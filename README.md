## Лабораторная работа по дисциплине информационные системы аэрокосмических комплексов
### Акимов В.Н. M30-312Б-18

Главный исполняемый файл - main.py, внутри комментарии ко всем функциям.
Файл city_data.xml генерируется автоматически при выполнении программы.

Принцип работы текущей версии (без метода из статьи):
1. На координатной плоскости широты и долготы откладываем два отрезка из COORD_UL к точке COORD_UR и COORD_LL.
2. Откладываем от заданной точки города вертикальную прямую к первому отрезку параллельно ко второму и ко второму парралельно к первому (строим крест).
3. Находим отношение длины отсеченных относительно точки COORD_UL отрезки ко всей длинне отрезков.
4. Откладываем эти же отношения на плоскости пиксельного изображения на сторонах заданного прямоугольника изображения снимка.
5. Из точки на верхней стороне прямоугольника откладываем прямую параллельно левой стороне прямоугольника и 
   из точки на левой стороне прямоугольника вторую прямую параллельно верхней стороне прямоугольника (строим крест).
6. На пересечении прямых будет лежать искомая точка пикселя, соответствующая заданной широте и долготе.
