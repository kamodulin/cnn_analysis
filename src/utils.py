import sys

def progress_bar(percent, **kwargs):
    bar_length = 40
    x = int(bar_length * percent)
    y = bar_length - x
    bar = f"[{'=' * x}{' ' * y}] {percent*100:>3.0f}%"
    for key, value in kwargs.items():
        bar += f" - {key}: {value:.3f}"
    sys.stdout.write("\r")
    sys.stdout.write(bar)
    sys.stdout.flush()