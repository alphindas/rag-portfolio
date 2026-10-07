# Widget Framework

Widget Framework is a lightweight tool for building reactive UI components without a build step.

## Installation

Install via pip.

### Requirements

You need Python 3.9 or later, and a modern browser for the dev server.

### Steps

Run `pip install widgetframework` in your terminal. Then verify the install by running `widgetframework --version`.

## Usage

### Basic Example

Here is a minimal example showing how to create a widget.

```python
from widgetframework import Widget

class Counter(Widget):
    def __init__(self):
        self.count = 0

    def render(self):
        return f"<button>{self.count}</button>"

    def on_click(self):
        self.count += 1
        self.rerender()
```

### Advanced Configuration

Widgets can be configured with a settings dict passed at construction time. Supported keys include `theme`, `debounce_ms`, and `lazy_render`. Setting `lazy_render` to True defers the initial render until the widget enters the viewport, which is useful for long pages with many widgets.

## Troubleshooting

If the dev server fails to start, check that port 8080 is free. If widgets fail to rerender, ensure you are calling `self.rerender()` and not mutating state directly without triggering a rerender.
