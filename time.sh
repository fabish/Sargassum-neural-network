#!/bin/bash
#Mide el tiempo de ejecucion de la red neuronal
function timef() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))

    if [ $total_seconds -gt 3599 ];  then
        printf "%02dh %02dm %02ds\n" $hours $minutes $seconds
        return
    fi

    if [ $total_seconds -gt 59 ];  then
        printf "%02dm %02ds\n" $minutes $seconds
        return
    fi

    printf "%02ds\n" $seconds
}

# Ejemplo de medición de tiempo


# ... ejecutar tarea o programa ...


echo $elapsed_seconds