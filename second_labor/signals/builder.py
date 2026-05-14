from .hawkes import HawkesSignal


class Signal:
    def __init__(self, observer, hawkes: HawkesSignal) -> None:
        self.hawkes = hawkes
        observer.subscribe([], self.hawkes.measure)
