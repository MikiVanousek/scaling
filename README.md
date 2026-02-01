## How to Run on Run:AI
Fill tokens in secret.yaml
kubectl apply -f secret.yaml


## Dataset Creation
It takes around 12 hours to create 1M row dataset and around 13 hours to subhchunk it. The latter process takes up to 35GB of RAM.
```
pixi run python notebooks/generate_ultrafineweb.py
```

```
pixi run python notebooks/subchunk_ultrafineweb.py --target_length 2048
```