class Category:

    # need to store all the events in this category, depending on the process from which they came
    def __init__(self, name):
        self.name = name
        self.event_content = {}
        self.weight_content = {}

    def add_events(self, events, weights, process):
        if len(events) != len(weights):
            raise Exception("Need to have exactly one weight per event!")

        if not process in self.event_content:
            self.event_content[process] = events
            self.weight_content[process] = weights
        else:
            self.event_content[process] = np.append(self.event_content[process], events, axis = 0)
            self.weight_content[process] = np.append(self.weight_content[process], weights, axis = 0)
