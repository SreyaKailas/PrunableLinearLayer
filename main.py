from train import train_model
from analyze import evaluate_and_plot

if __name__ == "__main__":
    # Test low, medium, and high Lambda values
    lambda_values = [1e-5, 1e-4, 1e-3]
    summary = []

    for l in lambda_values:
        model, test_loader = train_model(l, epochs=5) # 5 epochs for quick demo
        acc, sp = evaluate_and_plot(model, test_loader, l)
        summary.append((l, acc, sp))

    print("\n" + "="*30)
    print("FINAL SUMMARY TABLE")
    print("="*30)
    print(f"{'Lambda':<10} | {'Accuracy':<10} | {'Sparsity (%)':<10}")
    for l, acc, sp in summary:
        print(f"{l:<10.1e} | {acc:<10.2f}% | {sp:<10.2f}%")
