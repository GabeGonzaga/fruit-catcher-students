#!/usr/bin/env bash

# train.sh
# Executa python main.py -t 10 vezes, avisa e mostra o tempo de cada execução.

TOTAL=10

for i in $(seq 1 $TOTAL); do
  echo "=== Iniciando treino $i de $TOTAL ==="
  
  # Marca início
  start_time=$(date +%s)
  
  # Executa treino
  python main.py -t
  
  # Marca fim e calcula duração
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  
  echo ">>> Treino $i concluído em ${duration}s! <<<"
  
  # --- Opcional: notificação macOS ---
  # osascript -e "display notification \"Treino $i de $TOTAL concluído em ${duration}s.\" with title \"Fruit Catcher AI\""

  # --- Opcional: som de alerta ---
  # echo -ne "\a"
  
  echo
done

echo "Todos os $TOTAL treinos foram executados."