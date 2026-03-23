import os
import json
import subprocess
import config

def run_experiment(name, env_vars):
    print(f"Running experiment: {name}")
    env = os.environ.copy()
    env.update(env_vars)
    
    # Run the training script
    subprocess.run(["python", "run.py"], env=env, check=True)
    
    # After run.py finishes, it saves metrics to metrics.json
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            return metrics
    else:
        print(f"Warning: {metrics_path} not found for {name}")
        return None

def main():
    results = []
    
    # --- Dice Loss Ablation ---
    # Base: use_fgm=False, use_dice_loss=False
    base_metrics = run_experiment("Base (No Dice, No FGM)", {
        "BERT_LSTM_CRF_USE_DICE_LOSS": "0",
        "BERT_LSTM_CRF_USE_FGM": "0"
    })
    if base_metrics:
        base_metrics["Experiment"] = "Base (No Dice, No FGM)"
        results.append(base_metrics)

    # +Dice: use_dice_loss=True, dice_loss_weight=0.5
    dice_metrics = run_experiment("+Dice (weight=0.5)", {
        "BERT_LSTM_CRF_USE_DICE_LOSS": "1",
        "BERT_LSTM_CRF_DICE_LOSS_WEIGHT": "0.5",
        "BERT_LSTM_CRF_USE_FGM": "0"
    })
    if dice_metrics:
        dice_metrics["Experiment"] = "+Dice (weight=0.5)"
        results.append(dice_metrics)
        
    # --- FGM Hyperparameter Sensitivity ---
    for eps in [0.5, 1.0, 1.5, 2.0]:
        name = f"FGM (eps={eps})"
        metrics = run_experiment(name, {
            "BERT_LSTM_CRF_USE_DICE_LOSS": "0",
            "BERT_LSTM_CRF_USE_FGM": "1",
            "BERT_LSTM_CRF_FGM_EPSILON": str(eps)
        })
        if metrics:
            metrics["Experiment"] = name
            results.append(metrics)
            
    # Save results to CSV
    if results:
        import csv
        csv_file = "ablation_results.csv"
        keys = ["Experiment", "test_f1", "test_precision", "test_recall"] 
        # Add any other keys found in metrics
        for row in results:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
                    
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_file}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
