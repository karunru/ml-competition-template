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
cp ../ml-competition-template/.dockerignore .dockerignore
cp ../ml-competition-template/.gitignore .gitignore
cp ../ml-competition-template/Dockerfile Dockerfile
cp ../ml-competition-template/docker-compose.yml docker-compose.yml

# .gitkeep
touch features/.gitkeep
touch notebooks/.gitkeep
touch others/.gitkeep
touch output/.gitkeep
touch log/.gitkeep
