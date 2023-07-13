# directory
mkdir input
mkdir features
mkdir config
mkdir notebooks
mkdir others
mkdir output
mkdir log

# template
cp -r ../ml-competition-template/config/examples config/
cp -r ../ml-competition-template/input/* input/
cp -r ../ml-competition-template/src ./
cp ../ml-competition-template/main.py main.py
cp ../ml-competition-template/seed_averaging.py seed_averaging.py
cp ../ml-competition-template/make_seed_average_configs.py make_seed_average_configs.py
cp ../ml-competition-template/.gitignore .gitignore
cp ../ml-competition-template/pyproject.toml pyproject.toml

# .gitkeep
touch features/.gitkeep
touch notebooks/.gitkeep
touch others/.gitkeep
touch output/.gitkeep
touch log/.gitkeep

# idea settings
project_name=$(basename `pwd`)
cp -r ../ml-competition-template/.idea ./
mv .idea/ml-competition-template.iml .idea/"${project_name}.iml"
sed -i -e "s/ml\-competition\-template/${project_name}/g" .idea/modules.xml
sed -i -e "s/ml\-competition\-template/${project_name}/g" .idea/deployment.xml
sed -i -e "s/ml\-competition\-template/${project_name}/g" .idea/remote-mappings.xml

# poetry setting
sed -i -e "s/ml\-competition\-template/${project_name}/g" pyproject.toml
