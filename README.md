# Dark-Channel-Prior
### Технические требования

Для запуска скриптов на python:
	python >= 3.7
	

### Используемые скрипты на python

Перед использованием скриптов стоит установить зависимости:

```console
    pip install -r requirements.txt
```

Сами скрипты:

    * *diode_sampler.py* - скрипт, который для outdoor директории датасета DIODE[2] создает в выходной директории две папки, одна из которых - с изображениями, другая - с картами глубины, сохраненными в виде серых изображений. В [2] использовался сенсор с диапазоном в 0.5-350м, поэтому для неба и некоторый движущихся объектов значения получились невалидными. Для этих пикселей в картах глубины сделал значение в 2 раза больше максимальной глубины. Картинки и карты глубины для аугментации были сделаны из сплита DIODE для валидации.

### Тестирование

Приложение собрано и протестировано на:

1. ###### Linux

2. ###### Windows:

    *Visual Studio Code*:

        Генератор CMake - "Visual Studio 15 2017"

        CMake - 3.20.1

        MSVC - 19.16.27045.0

        OpenCV - 4.5.5

### Ссылки:

2. Vasiljevic I. et al. Diode: A dense indoor and outdoor depth dataset //arXiv preprint arXiv:1908.00463. – 2019.