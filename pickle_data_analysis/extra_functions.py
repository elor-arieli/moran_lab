def get_events_between_times(event_times, start_time, stop_time):
    return [i for i in event_times if start_time < i < stop_time]