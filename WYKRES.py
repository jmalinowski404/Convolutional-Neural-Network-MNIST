import matplotlib.pyplot as plt
import re
from collections import defaultdict

file_path = 'train_log.txt'

max_batches_per_epoch = 937
window_size = 50

buckets = defaultdict(list)
epochs_list = set()

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'\[(\d+),\s*(\d+)\] loss:\s*([\d\.\-eE]+)', line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss = float(match.group(3))

                epochs_list.add(epoch)

                global_step = (epoch - 1) * max_batches_per_epoch + batch

                bucket_index = global_step // window_size
                buckets[bucket_index].append(loss)

    if not buckets:
        print("Nie znaleziono pasujących danych w pliku. Sprawdź format logów.")
    else:
        x_vals = []
        y_vals = []

        for bucket_idx in sorted(buckets.keys()):
            avg_loss = sum(buckets[bucket_idx]) / len(buckets[bucket_idx])

            avg_global_step = (bucket_idx * window_size) + (window_size / 2)
            fractional_epoch = (avg_global_step / max_batches_per_epoch) + 1

            x_vals.append(fractional_epoch)
            y_vals.append(avg_loss)

        plt.figure(figsize=(10, 6))

        plt.plot(x_vals, y_vals, linestyle='-', color='b', linewidth=1.5, alpha=0.9)

        plt.title('Wykres błedu do epok (średnia co 50 batchy)')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(sorted(list(epochs_list)))

        plt.tight_layout()

        plt.savefig('loss.png')

except FileNotFoundError:
    print(f"Błąd: Nie można znaleźć pliku '{file_path}'.")