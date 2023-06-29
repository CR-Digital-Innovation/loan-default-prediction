createEnv:
	pip install virtualenv & \
		virtualenv ~/.venv

activateEnv:
	source ~/.venv/bin/activate

newEnv:
	createEnv activateEnv

install:
	pip install --upgrade pip & \
		pip install -r requirements-dev.txt

test:
	python pytest --nbval src/*.ipynb

lint:
	pylint --disable=R,C src

format:
	black src

docker-build:
	docker-compose build

docker-deploy:
	docker-compose up --build

docker-destroy:
	docker-compose down --rmi all

all:
	install lint