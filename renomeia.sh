#!/bin/bash

diretorio="$1"
novo_nome="$2"

if [ -z "$diretorio" ] || [ -z "$novo_nome" ]; then
    echo "Uso: $0 <diretorio> <novo_nome>"
    exit 1
fi

if [ ! -d "$diretorio" ]; then
    echo "Erro: diretório não encontrado: $diretorio"
    exit 1
fi

i=0

for file in "$diretorio"/*; do
    [ -f "$file" ] || continue

    ext="${file##*.}"

    mv -- "$file" "$diretorio/${novo_nome}${i}.${ext}"

    ((i++))
done
