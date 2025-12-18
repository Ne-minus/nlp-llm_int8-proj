#!/bin/bash

# Массив с размерами
sizes=(0.6 1.7 4 8)

# Цикл по размерам и запуск Python скрипта
for size in "${sizes[@]}"; do
    python quant_6.py --size $size --mode full
done
