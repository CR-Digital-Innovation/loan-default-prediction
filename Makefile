install:
	pip install --upgrade pip & \
		pip install -r requirements.txt

test:
	python -m pytest --nbval src/*.ipynb

lint:
	pylint --disable=R,C src

all:
	install test