IMAGE_NAME = ml-test-task
CONTAINER_NAME = ml-test-task-app

build:
	echo "Создание/обновление docker-образа"
	docker build -t $(IMAGE_NAME) .

train: build
	echo "Запуск процесса обучения модели"
	docker run -it --rm -v .:/usr/src/app --name $(IMAGE_NAME) $(IMAGE_NAME) python train.py

test: build
	echo "Запуск процесса тестирования модели"
	docker run -it --rm -v .:/usr/src/app --name $(IMAGE_NAME) $(IMAGE_NAME) python test.py

fullrun: train test