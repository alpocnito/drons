# Мотивация
Определение положения беспилотников с помощью микрофонов. Микрофоны записывают звук, над звуком произоводится wavelet анализ, результаты отправляются в нейросеть, которая определяет есть ли беспилотник.

# Описание данных

В папке data находятся аудиозаписи и текстовое представление аудио.

LmicX-14-04-33.wav - запись с левого микрофона
RmicX-14-04-33.wav - запись с правого микрофона
дроны.txt - текстовые представление аудио, где по столбцам:
- unixtime в секундах 
- данные с левого микрофона
- данные с правого микрофона
- угол по горизонтали
- угол по вертикали, 180 - микрофоны направлены вниз, 0 - вверх

Частота дискретизации данных в текстовом представлении - 44100 Гц.

# Описание задачи
 
1. Разбить аудио на промежутки 2-5 сек
2. Посчитать среднее по промежутку, вычесть из промежутка посчитанное среднее. Это нужно для исключения шумов ветра 
3. Произвести wavelet анализ промежутков, определить удобный базис и необходимую частоту дискретизации, отметить на wavelet-графе спектры беспилотников. Эти wavelet-графы отправятся в нейросеть.

Примерные времена:

=== минута 1 ===

подлет коптеров
18-20, 21-25, 26-29, 50-51

пролет коптеров
20, 29, 51, 55-56, 59-60

ветер
10, 14, 26-27, 30-50


=== минута 2 ===

подлет коптера
30-32

удаление коптеров
0-5, 34-40

пролет коптеров
32, 34

голоса
0-30, 40-50

сильный ветер
53-60

=== минута 3 ===

подлет коптеров
3-5, 7-9, 46-50
пролет коптеров
5-6, 9, 50-53
удаление коптеров
9-25, 53-60

голоса
0-1, 19, 24-26, 29-32

ветер
12-15, 24, 29, 45, 49-50, 54-60


=== минута 4 ===

ветер
1-10, 12-36

взрыв
2, 6, 15, 18, 20, 37, 39, 43, 46, 47, 50

голоса
8-9, 37-42, 48-53

минута 5
0-1, 10-16, 37, 50-60 - ветер
58-60 - сильный ветер
1, 5, 11, 12, 26, 28, 32 - взрыв

минута 6
0-52 - ветер
31 - сильный ветер
44, 49 - завывающий ветер
2-7, 15-30, 49- - голоса
56-57 - 2 взрыва
