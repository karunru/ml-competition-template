import gc
import itertools
from typing import Dict, List, Union

import cudf
import gensim.downloader
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch
import transformers
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.preprocessing.text import text_to_word_sequence
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from src.utils import timer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertTokenizer, pipeline
from xfeat.types import XDataFrame
from xfeat.utils import is_cudf

"""https://github.com/okotaku/pet_finder/blob/master/code/all_tools.py
"""


# ===============
# Feature Engineering
# ===============
class GroupbyTransformer:
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict["key"]
        if "var" in p_dict.keys():
            var = p_dict["var"]
        else:
            var = self.var
        if "agg" in p_dict.keys():
            agg = p_dict["agg"]
        else:
            agg = self.agg
        if "on" in p_dict.keys():
            on = p_dict["on"]
        else:
            on = key
        return key, var, agg, on

    def _aggregate(self, dataframe):
        with timer("aggregate"):
            self.features = []
            for param_dict in tqdm(self.param_dict):
                key, var, agg, on = self._get_params(param_dict)
                all_features = list(set(key + var))
                new_features = self._get_feature_names(key, var, agg)
                features = (
                    dataframe[all_features].groupby(key)[var].agg(agg).reset_index()
                )
                features.columns = key + new_features
                self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        with timer("merge"):
            for param_dict, features in tqdm(
                zip(self.param_dict, self.features), total=len(self.features)
            ):
                key, var, agg, on = self._get_params(param_dict)
                if merge:
                    if is_cudf(dataframe):
                        dataframe = cudf.merge(dataframe, features, how="left", on=on)
                    else:
                        dataframe = dataframe.merge(features, how="left", on=on)
                else:
                    new_features = self._get_feature_names(key, var, agg)
                    dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ["_".join([a, v, "groupby"] + key) for v in var for a in _agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()


class DiffGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    if not isinstance(a, str):
                        new_feature = "_".join(["diff", a.__name__, v, "groupby"] + key)
                        base_feature = "_".join([a.__name__, v, "groupby"] + key)
                    else:
                        new_feature = "_".join(["diff", a, v, "groupby"] + key)
                        base_feature = "_".join([a, v, "groupby"] + key)
                    print(new_feature)
                    dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ["_".join(["diff", a, v, "groupby"] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    if not isinstance(a, str):
                        new_feature = "_".join(
                            ["ratio", a.__name__, v, "groupby"] + key
                        )
                        base_feature = "_".join([a.__name__, v, "groupby"] + key)
                    else:
                        new_feature = "_".join(["ratio", a, v, "groupby"] + key)
                        base_feature = "_".join([a, v, "groupby"] + key)
                    print(new_feature)
                    dataframe[new_feature] = dataframe[v] / dataframe[base_feature]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ["_".join(["ratio", a, v, "groupby"] + key) for v in var for a in _agg]


class CategoryVectorizer:
    def __init__(
        self,
        categorical_columns,
        n_components,
        vectorizer=CountVectorizer(),
        transformer=LatentDirichletAllocation(),
        name="CountLDA",
    ):
        self.categorical_columns: List[str] = categorical_columns
        self.n_components: int = n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name: str = name + str(self.n_components)
        self.columns: List[str] = []

    def transform(self, dataframe: XDataFrame) -> XDataFrame:
        features = []
        for (col1, col2) in tqdm(self.get_column_pairs()):
            try:
                sentence = self.create_word_list(dataframe, col1, col2)
                sentence = self.vectorizer.fit_transform(sentence)
                feature = self.transformer.fit_transform(sentence)
                feature = self.get_feature(
                    dataframe, col1, col2, feature, name=self.name
                )
                features.append(feature)
            except Exception:
                pass

        features = (
            pd.concat(features, axis=1)
            if isinstance(dataframe, pd.DataFrame)
            else cudf.concat(features, axis=1)
        )
        return features

    def create_word_list(self, dataframe, col1, col2) -> List:
        col1_size = int(dataframe[col1].max() + 1)
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(dataframe[col1].to_array(), dataframe[col2].to_array()):
            col2_list[int(val1)].append(col2 + str(val2))
        return [" ".join(map(str, ls)) for ls in col2_list]

    def get_feature(self, dataframe, col1, col2, latent_vector, name="") -> XDataFrame:
        features = np.zeros(shape=(len(dataframe), self.n_components), dtype=np.float32)
        self.columns = [
            "_".join([name, col1, col2, str(i)]) for i in range(self.n_components)
        ]
        for i, val1 in enumerate(dataframe[col1].to_pandas().fillna(0).astype(int)):
            features[i, : self.n_components] = latent_vector[val1]

        return (
            pd.DataFrame(data=features, columns=self.columns)
            if isinstance(dataframe, pd.DataFrame)
            else cudf.DataFrame(data=features, columns=self.columns)
        )

    def get_column_pairs(self) -> List[tuple]:
        return [
            (col1, col2)
            for col1, col2 in itertools.product(self.categorical_columns, repeat=2)
            if col1 != col2
        ]

    def get_numerical_features(self) -> List:
        return self.columns


class SinCos:
    def __init__(self, feature_name, period):
        """
        input
        ---
        feature_name(str): name of feature
        period(int): period of feature
        """
        self.feature_name = feature_name
        self.period = period

    def create_features(self, df):
        df["{}_sin".format(self.feature_name)] = np.sin(
            2 * np.pi * df[self.feature_name] / self.period
        )
        df["{}_cos".format(self.feature_name)] = np.cos(
            2 * np.pi * df[self.feature_name] / self.period
        )
        new_cols = ["{}_{}".format(self.feature_name, key) for key in ["sin", "cos"]]

        return df, new_cols


class Frequency:
    def __init__(self, categorical_columns):
        """
        input
        ---
        categorical_columns(list): categorical columns
        """
        self.categorical_columns = categorical_columns

    def create_features(self, df):
        new_cols = []
        for col in self.categorical_columns:
            fname = "{}_Frequency".format(col)
            df[fname] = df.groupby(col)[col].transform("count") / len(df)
            new_cols.append(fname)

        return df, new_cols


class TargetEncoder:
    def __init__(self, n_splits: int = 5, random_state: int = 128):
        self.class_dict: Dict[str, List[float]] = {}
        self.column = ""
        self.n_splits = n_splits
        self.random_state = random_state

    def transform(self, X_: pd.DataFrame) -> np.ndarray:
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        X = X_.copy()
        X = X.reset_index(drop=True)
        converted = np.zeros(len(X))
        for i, (_, v_idx) in enumerate(kf.split(X)):
            converted[v_idx] = X.loc[v_idx, self.column].map(
                lambda x: self.class_dict[x][i]
            )
        return converted

    def fit_transform(
        self, X_: pd.DataFrame, y: Union[pd.Series, np.ndarray], column: str
    ) -> np.ndarray:
        self.column = column
        uniq_class = X_[column].unique()
        for c in uniq_class:
            self.class_dict[c] = []
        kf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        X = X_.copy()
        X = X.reset_index(drop=True)
        yy = y.values if isinstance(y, pd.Series) else y
        converted = np.zeros(len(X))
        # import pdb
        # pdb.set_trace()
        for t_idx, v_idx in kf.split(X, y):
            X_t = X.loc[t_idx, column]
            y_t = yy[t_idx]
            X_v = X.loc[v_idx, column]
            cvtd = converted[v_idx]

            for c in uniq_class:
                target_mean = y_t[X_t == c].mean()
                self.class_dict[c].append(target_mean)
                cvtd[X_v == c] = target_mean
            converted[v_idx] = cvtd
        return converted


# ===============
# text feature extraction
# https://github.com/Ynakatsuka/kaggle_utils/blob/master/kaggle_utils/features/text.py
# ===============


class BasicTextFeatureTransformer:
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def _get_features(self, dataframe, column):
        dataframe[column + "_num_chars"] = dataframe[column].apply(len)
        dataframe[column + "_num_capitals"] = dataframe[column].apply(
            lambda x: sum(1 for c in x if c.isupper())
        )
        dataframe[column + "_caps_vs_length"] = (
            dataframe[column + "_num_chars"] / dataframe[column + "_num_capitals"]
        )
        dataframe[column + "_num_exclamation_marks"] = dataframe[column].apply(
            lambda x: x.count("!")
        )
        dataframe[column + "_num_question_marks"] = dataframe[column].apply(
            lambda x: x.count("?")
        )
        dataframe[column + "_num_punctuation"] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in ".,;:")
        )
        dataframe[column + "_num_symbols"] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in "*&$%")
        )
        dataframe[column + "_num_words"] = dataframe[column].apply(
            lambda x: len(x.split())
        )
        dataframe[column + "_num_unique_words"] = dataframe[column].apply(
            lambda x: len(set(w for w in x.split()))
        )
        dataframe[column + "_words_vs_unique"] = (
            dataframe[column + "_num_unique_words"] / dataframe[column + "_num_words"]
        )
        dataframe[column + "_num_smilies"] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in (":-)", ":)", ";-)", ";)"))
        )
        return dataframe

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[self.text_columns] = (
            dataframe[self.text_columns].astype(str).fillna("missing")
        )
        for c in self.text_columns:
            dataframe = self._get_features(dataframe, c)
        return dataframe


class TextVectorizer:
    def __init__(
        self,
        text_columns,
        vectorizer=CountVectorizer(),
        transformer=TruncatedSVD(n_components=64, random_state=1031),
        name="count_svd",
    ):
        self.text_columns = text_columns
        self.n_components = transformer.n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[self.text_columns] = (
            dataframe[self.text_columns].astype(str).fillna("missing")
        )
        features = []
        for c in self.text_columns:
            sentence = self.vectorizer.fit_transform(dataframe[c])
            feature = self.transformer.fit_transform(sentence)
            feature = pd.DataFrame(
                feature,
                columns=[
                    c + "_" + self.name + f"_{i:03}" for i in range(self.n_components)
                ],
            )
            features.append(feature)
        dataframe = pd.concat([dataframe] + features, axis=1)
        return dataframe


class Doc2VecFeatureTransformer:
    def __init__(
        self,
        text_columns,
        name="doc2vec",
    ):
        self.text_columns = text_columns
        self.name = name

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.features = []
        for c in self.text_columns:
            texts = dataframe[c].astype(str)
            corpus = [
                TaggedDocument(words=text, tags=[i]) for i, text in enumerate(texts)
            ]
            model = Doc2Vec(documents=corpus)
            result = np.array([model.infer_vector(text.split(". ")) for text in texts])
            features = pd.DataFrame(
                {
                    f"{c}_{self.name}_mean": np.mean(result, axis=1),
                    f"{c}_{self.name}_median": np.median(result, axis=1),
                    f"{c}_{self.name}_sum": np.sum(result, axis=1),
                    f"{c}_{self.name}_max": np.max(result, axis=1),
                    f"{c}_{self.name}_min": np.min(result, axis=1),
                    f"{c}_{self.name}_var": np.var(result, axis=1),
                }
            )
            self.features.append(features)
        dataframe = pd.concat([dataframe] + self.features, axis=1)
        return dataframe


class W2VFeatureTransformer:
    """
    from gensim.models import FastText, word2vec, KeyedVectors

    model = word2vec.Word2Vec.load('../data/w2v.model')
    # model = KeyedVectors.load_word2vec_format(path, binary=True)
    """

    ps = nltk.stem.PorterStemmer()
    lc = nltk.stem.lancaster.LancasterStemmer()
    sb = nltk.stem.snowball.SnowballStemmer("english")

    def __init__(
        self,
        text_columns,
        model=None,
        transformer=TruncatedSVD(n_components=64, random_state=1031),
        name="w2v",
    ):
        self.text_columns = text_columns
        self.model = (
            model
            if model is not None
            else gensim.downloader.load("word2vec-google-news-300")
        )
        self.transformer = transformer
        self.n_components = transformer.n_components
        self.name = name

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.features = []
        for c in self.text_columns:
            texts = dataframe[c].astype(str)
            texts = [text_to_word_sequence(text) for text in texts]
            result = []
            for text in texts:
                n_skip = 0
                vec = np.zeros(self.model.vector_size)
                for n_w, word in enumerate(text):
                    if self.model.__contains__(word):
                        vec = vec + self.model[word]
                        continue
                    word_ = word.upper()
                    if self.model.__contains__(word_):
                        vec = vec + self.model[word_]
                        continue
                    word_ = word.capitalize()
                    if self.model.__contains__(word_):
                        vec = vec + self.model[word_]
                        continue
                    word_ = self.ps.stem(word)
                    if self.model.__contains__(word_):
                        vec = vec + self.model[word_]
                        continue
                    word_ = self.lc.stem(word)
                    if self.model.__contains__(word_):
                        vec = vec + self.model[word_]
                        continue
                    word_ = self.sb.stem(word)
                    if self.model.__contains__(word_):
                        vec = vec + self.model[word_]
                        continue
                    else:
                        n_skip += 1
                        continue
                vec = vec / (n_w - n_skip + 1)
                result.append(vec)
            result = pd.DataFrame(
                self.transformer.fit_transform(np.nan_to_num(result)),
                columns=[f"{c}_{self.name}_{i:03}" for i in range(self.n_components)],
            )
            self.features.append(result)
        dataframe = pd.concat([dataframe] + self.features, axis=1)
        return dataframe


class USEFeatureTransformer:
    """
    Example
    -------
    urls = [
        'https://tfhub.dev/google/universal-sentence-encoder/4',
    ]
    """

    def __init__(
        self,
        text_columns,
        urls=["https://tfhub.dev/google/universal-sentence-encoder/4"],
        transformer=TruncatedSVD(n_components=64, random_state=1031),
        name="use",
    ):
        self.text_columns = text_columns
        self.urls = urls
        self.transformer = transformer
        self.n_components = transformer.n_components
        self.name = name

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.features = []
        for url in self.urls:
            model_name = url.split("/")[-2]
            embed = hub.load(url)
            for c in self.text_columns:
                texts = dataframe[c].astype(str)
                result = embed(texts).numpy()
                result = pd.DataFrame(
                    self.transformer.fit_transform(np.nan_to_num(result)),
                    columns=[
                        f"{c}_{self.name}_{i:03}" for i in range(self.n_components)
                    ],
                )
                self.features.append(result)
        dataframe = pd.concat([dataframe] + self.features, axis=1)
        tf.keras.backend.clear_session()
        del embed
        gc.collect()
        return dataframe


class BERTFeatureTransformer:
    """
    Reference
    ---------
    https://huggingface.co/transformers/pretrained_models.html

    Example
    -------
    """

    def __init__(
        self,
        text_columns,
        model_names=["bert-base-uncased"],
        batch_size=8,
        device=-1,
        transformer=TruncatedSVD(n_components=64, random_state=1031),
    ):
        self.text_columns = text_columns
        self.model_names = model_names
        self.batch_size = batch_size
        self.device = device
        self.transformer = transformer
        self.n_components = transformer.n_components

    def transform(self, dataframe):
        self.features = []
        for model_name in self.model_names:
            model = pipeline("feature-extraction", device=self.device, model=model_name)
            for c in self.text_columns:
                texts = dataframe[c].astype(str).tolist()
                result = []
                for i in tqdm(range(np.ceil(len(texts) / self.batch_size).astype(int))):
                    result.append(
                        np.max(
                            model(
                                texts[
                                    i
                                    * self.batch_size : min(
                                        len(texts), (i + 1) * self.batch_size
                                    )
                                ]
                            ),
                            axis=1,
                        )
                    )
                result = np.concatenate(result, axis=0)
                result = pd.DataFrame(
                    self.transformer.fit_transform(np.nan_to_num(result)),
                    columns=[
                        f"{c}_{model_name}_{i:03}" for i in range(self.n_components)
                    ],
                )
                self.features.append(result)
        dataframe = pd.concat([dataframe] + self.features, axis=1)
        return dataframe


class GroupedCategoriesWord2VecFeatureTransformer:
    def __init__(
        self,
        listed_cat_columns,
        w2v_size="auto",
        svd_size=-1,
        name="grouped_word2vec",
    ):
        self.listed_cat_columns = listed_cat_columns
        self.name = name
        self.w2v_size = w2v_size
        self.transformer = (
            TruncatedSVD(n_components=svd_size, random_state=1031)
            if svd_size > 1
            else -1
        )
        self.agg_dict = {
            "mean": np.mean,
            "median": np.median,
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
            "std": np.std,
        }

    def transform(self, grouped_dataframe: pd.DataFrame) -> pd.DataFrame:
        self.features = []
        for c in self.listed_cat_columns:
            self.w2v_size = (
                int(
                    len(
                        set(
                            [
                                item
                                for sublist in grouped_dataframe[c].values.tolist()
                                for item in sublist
                            ]
                        )
                    )
                    / 4
                )
                if self.w2v_size == "auto"
                else self.w2v_size
            )

            w2v_model = word2vec.Word2Vec(
                grouped_dataframe[c].values.tolist(),
                size=self.w2v_size,
                min_count=1,
                window=1,
                iter=100,
            )
            features = pd.DataFrame()

            for func in self.agg_dict.keys():
                features[
                    [
                        f"{c}_{self.name}_{func}_{i}"
                        for i in range(w2v_model.vector_size)
                    ]
                ] = pd.DataFrame(
                    np.vstack(
                        [
                            x
                            for x in grouped_dataframe[c].progress_apply(
                                lambda x: self.agg_dict[func](
                                    [w2v_model.wv[e] for e in x], axis=0
                                )
                            )
                        ]
                    )
                )

            if self.transformer != -1:
                for func in self.agg_dict.keys():
                    features[
                        [
                            f"{c}_{self.name}_{func}_svd_{i}"
                            for i in range(self.transformer.n_components)
                        ]
                    ] = pd.DataFrame(
                        np.vstack(
                            self.transformer.fit_transform(
                                np.nan_to_num(
                                    features[
                                        [
                                            f"{c}_{self.name}_{func}_{i}"
                                            for i in range(w2v_model.vector_size)
                                        ]
                                    ].values
                                )
                            ),
                        )
                    )
                    features = features.drop(
                        columns=[
                            f"{c}_{self.name}_{func}_{i}"
                            for i in range(w2v_model.vector_size)
                        ]
                    )

            self.features.append(features)
        dataframe = pd.concat([grouped_dataframe] + self.features, axis=1)
        return dataframe


# referred to : https://github.com/xhlulu/dl-translate/blob/main/dl_translate/_translation_model.py
class TranslationModel:
    def __init__(
        self,
        model_or_path: str = "Helsinki-NLP/opus-mt-nl-en",
        tokenizer_path: str = None,
        device: str = "auto",
        model_options: dict = None,
        tokenizer_options: dict = None,
        model_to_half: bool = False,
    ):
        """
        *Instantiates a multilingual transformer model for translation.*
        {{params}}
        {{model_or_path}} The path or the name of the model. Equivalent to the first argument of `AutoModel.from_pretrained()`.
        {{tokenizer_path}} The path to the tokenizer. By default, it will be set to `model_or_path`.
        {{device}} "cpu", "gpu" or "auto". If it's set to "auto", will try to select a GPU when available or else fall back to CPU.
        {{model_options}} The keyword arguments passed to the model, which is a transformer for conditional generation.
        {{tokenizer_options}} The keyword arguments passed to the model's tokenizer.
        {{model_to_half}} Whether to halve the model.
        """
        self.model_or_path = model_or_path
        self.device = self._select_device(device)

        # Resolve default values
        tokenizer_path = tokenizer_path or self.model_or_path
        model_options = model_options or {}
        tokenizer_options = tokenizer_options or {}

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, truncation=True, model_max_length=512, **tokenizer_options
        )

        # Load the model either from a saved torch model or from transformers.
        self._transformers_model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_or_path, **model_options)
            .to(self.device)
            .eval()
        )
        if model_to_half:
            self._transformers_model = self._transformers_model.half()

    def translate(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 50,
    ) -> Union[str, List[str]]:
        """
        *Translates a string or a list of strings from a source to a target language.*
        {{params}}
        {{text}} The content you want to translate.
        {{batch_size}} The number of samples to load at once. If set to `None`, it will process everything at once.
        {{max_length}} The max number of translated sentences length
        Note:
        - A smaller value is preferred for `batch_size` if your (video) RAM is limited.
        """
        original_text_type = type(text)
        if original_text_type is str:
            text = [text]

        if batch_size is None:
            batch_size = len(text)

        data_loader = torch.utils.data.DataLoader(text, batch_size=batch_size)
        output_text = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )["input_ids"].to(self.device)

                generated_tokens = self._transformers_model.generate(
                    encoded, max_length=max_length, num_beams=4, early_stopping=True
                ).cpu()

                decoded = self._tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                output_text.extend(decoded)

        # If text: str and output_text: List[str], then we should convert output_text to str
        if original_text_type is str and len(output_text) == 1:
            output_text = output_text[0]

        return output_text

    def _select_device(self, device_selection):
        selected = device_selection.lower()
        if selected == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif selected == "cpu":
            device = torch.device("cpu")
        elif selected == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device(selected)

        return device


class BertSequenceVectorizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name,
            truncation=True,
            max_length=512,
        )
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name).half()
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence
                    ,truncation=True,
                    max_length=512,)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[: self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out["last_hidden_state"], bert_out["pooler_output"]

        if torch.cuda.is_available():
            return (
                seq_out[0][0].cpu().detach().numpy()
            )  # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()
