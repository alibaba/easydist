__pipeline_tracer_global = None


def get_pipeline_tracer():
    global __pipeline_tracer_global
    return __pipeline_tracer_global


def set_pipeline_tracer(tracer):
    global __pipeline_tracer_global
    __pipeline_tracer_global = tracer
