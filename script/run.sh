# Generation (level 1)
python3 script/generation_1.py --deployment_name gpt-4 \
                        --max_tokens 300 --temperature 0.0 --top_p 0.0 \
                        --data data/input/sample.jsonl \
                        --prompt_file prompt/generation_1.txt \
                        --seed_file prompt/seed_1.md \
                        --out_file data/output/generation_1.jsonl \
                        --topic_file data/output/generation_1.md \
                        --verbose True


# Assignment
python3 script/assignment.py --deployment_name gpt-3.5-turbo \
                        --max_tokens 300 --temperature 0.0 --top_p 0.0 \
                        --data data/input/sample.jsonl \
                        --prompt_file prompt/assignment.txt \
                        --topic_file data/output/generation_1.md \
                        --out_file data/output/assignment.jsonl \
                        --verbose True


# Self-correction
python3 script/correction.py --deployment_name gpt-3.5-turbo \
                        --max_tokens 300 --temperature 0.0 --top_p 0.0 \
                        --data data/output/assignment.jsonl \
                        --prompt_file prompt/correction.txt \
                        --topic_file data/output/generation_1.md \
                        --out_file data/output/assignment_corrected.jsonl \
                        --verbose True