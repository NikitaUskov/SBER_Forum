# SBER_Forum
Разработка системы, которая будет тестировать алгоритмы роботов, то есть их мозгов, то, как они думают, строят себе какие-то действия.
Итог - это программа, которая на вход получает правильный ответ, ответ робота, еще какие-то вводные темы, обстановка и само задание. И выдает, способен ли робот с таким планом выполнить задачу или где-то есть логическая ошибка. То есть итог это программа, которая позволяет тестировать мозги робота.
Удалось реализовать программу, которая получает все необходимые данные и выдает предсказания, подходит ли план робота под реальный план или там есть где-то логическая ошибка. Точность +- 74%.
Удалось реализовать три варианта программы. Один - без обработки, это сырой бейзлайн, который по всем данным классифицирует. Другой вариант - предобработка текста, его токенизация, его лимитизация и уже классификация обработанного текста. Именно он и был нашим финальным вариантом. Третий вариант это работа через языковую модель, которая получает на вход все данные, сама обрабатывает и выдает ответ, классификацию.
