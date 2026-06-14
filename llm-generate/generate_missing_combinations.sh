#!/usr/bin/env bash

models=$(ollama list | sed '1d' | cut -d' ' -f1 | grep -v -f exclude.txt)
prompts=$(ls prompts/ | rev | cut -d'.' -f2- | rev)

for model in $models ; do
    for prompt in $prompts ; do
        pattern="${model}-${prompt}"
        if ! ls output/ | grep -q "$pattern" ; then
            prompt_file=$(ls prompts/ | grep "${prompt}")

            MODEL="${model}" \
                 PROMPT_FILE=prompts/${prompt_file} \
                 OUTPUT_DIR=output/ python generate.py
        fi
    done
done
