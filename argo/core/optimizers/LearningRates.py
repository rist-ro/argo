import tensorflow as tf
import copy

EXPONENTIAL_DECAY = 'exponential_decay'

POLYNOMIAL_DECAY = 'polynomial_decay'

known_schedules = {
    # schedule as from magenta wavenet github
    "magenta_sched":  {
        0:      2e-4,
        90000:  4e-4 / 3,
        120000: 6e-5,
        150000: 4e-5,
        180000: 2e-5,
        210000: 6e-6,
        240000: 2e-6
    },
    "sinusoid_sched": {
        0:    2e-4,
        4000: 4e-4 / 3,
        7000: 6e-5
    }
}

known_schedules_short = {
    "magenta_sched":  "m",
    "sinusoid_sched": "s"
}

def check_decay_kwargs(kwargs):
    has_decay_epochs = "decay_epochs" in kwargs
    has_decay_steps = "decay_steps" in kwargs

    if has_decay_epochs and has_decay_steps:
        raise Exception("can only have decay_steps or decay_epochs in the arguments for the learning rate, not both.")

    return has_decay_epochs, has_decay_steps

def convert_decay_epochs_in_steps(kwargs, n_batchs_per_epoch):
    has_decay_epochs, has_decay_steps = check_decay_kwargs(kwargs)

    if has_decay_epochs:
        decay_epochs = kwargs.pop("decay_epochs")
        kwargs['decay_steps'] = decay_epochs * n_batchs_per_epoch

    return kwargs

def decay_id(kwargs):
    has_decay_epochs, has_decay_steps = check_decay_kwargs(kwargs)

    if has_decay_steps:
        return ".s" + "{:.0e}".format(kwargs["decay_steps"])
    elif has_decay_epochs:
        return ".e" + "{:.4g}".format(kwargs["decay_epochs"])

def process_learning_rate(learning_rate, global_step, n_batchs_per_epoch):
    """
    Args:
        global_step:
        learning_rate: can be either a number, a dictionary {"step": lr}, e.g. {100:0.1, 1000:0.001} or a particular keyword

    Returns:
        the processed learning rate, ready to be taken by the optimizer (if a schedule was requested returns a tf node)
    """

    lr = None

    if isinstance(learning_rate, str):
        if learning_rate in known_schedules:
            learning_rate = known_schedules[learning_rate]
        else:
            raise ValueError(
                "requested schedule: %s not found in known_schedules: %s" % (learning_rate, known_schedules))

    elif isinstance(learning_rate, (int, float)):
        lr = learning_rate

    elif isinstance(learning_rate, tuple):
        lr_min, lr_name, lr_kwargs = learning_rate
        lr_kwargs = copy.deepcopy(lr_kwargs)
        lr_kwargs = convert_decay_epochs_in_steps(lr_kwargs, n_batchs_per_epoch)
        lr_method = getattr(tf.train, lr_name)
        lr_kwargs.update({"global_step": global_step})
        lr = lr_min + lr_method(**lr_kwargs)

    # instantiate lr node if lr is None and learning_rate is a dict at this point
    if lr is None and isinstance(learning_rate, dict):
        if not 0 in learning_rate:
            raise ValueError(
                "learning rate schedule must specify, learning rate for step 0. Found schedule: %s" % learning_rate)

        lr = tf.constant(learning_rate[0])
        # THIS IS PASSED FROM OUTSIDE global_step = tf.train.get_or_create_global_step()
        for key, value in learning_rate.items():
            lr = tf.cond(
                tf.less(global_step, key), lambda: lr, lambda: tf.constant(value))
        tf.summary.scalar("learning_rate", lr)

    if lr is None:
        raise Exception("oops, something went wrong... could not process learning rate {}".format(str(learning_rate)))

    return tf.identity(lr, name="learning_rate")


def get_lr_id(learning_rate):
    """
    Args:
        learning_rate: can be either a number, a dictionary {"step": lr}, e.g. {100:0.1, 1000:0.001} or a particular keyword,
                    or a tuple (lr_min, lr_name, lr_kwargs)

    Returns:
        the id for the learning rate
    """

    _id = ""
    if isinstance(learning_rate, str):
        if learning_rate in known_schedules_short:
            _id += "S"
            _id += known_schedules_short[learning_rate]
        else:
            raise ValueError(
                "requested schedule: %s not found in known_schedules: %s" % (learning_rate, known_schedules_short))


    elif isinstance(learning_rate, (int, float)):
        _id += str(learning_rate)

    elif isinstance(learning_rate, tuple):
        lr_min, lr_name, lr_kwargs = learning_rate
        if lr_min > 0.:
            _id += lr_method_id(lr_name, lr_kwargs)
            _id += "m" + str(lr_min)

    elif isinstance(learning_rate, dict):
        keys=learning_rate.keys()
        val=learning_rate.values()
        name = []
        for k,v in zip(keys,val):
            name.append("{}r{}".format(k,v))
        n = "_".join(name)

        _id += n

    return _id

# lr_method_name_short = {
#     EXPONENTIAL_DECAY: "Ed",
#     POLYNOMIAL_DECAY:  "Pd"
# }

lr_method_name_short = {
    EXPONENTIAL_DECAY: "Ed",
    POLYNOMIAL_DECAY:  "Pd"
}


def lr_method_id(lr_name, lr_kwargs):
    """Creates the id for the learning rate method.
    """

    # listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))

    methodid = lr_method_name_short[lr_name]

    if lr_name == EXPONENTIAL_DECAY:
        methodid += ".i" + "{:.0e}".format(lr_kwargs["learning_rate"])
        methodid += decay_id(lr_kwargs)
        methodid += ".r" + "{}".format(lr_kwargs["decay_rate"])

    elif lr_name == POLYNOMIAL_DECAY:
        methodid += ".i" + "{:.0e}".format(lr_kwargs["learning_rate"])
        methodid += decay_id(lr_kwargs)
        methodid += ".r" + "{}".format(lr_kwargs["end_learning_rate"])
        methodid += ".p" + "{}".format(lr_kwargs["power"])

    else:
        print('----------------------')
        print('ERROR ', lr_name)
        raise ValueError("id rule for learning rate `%s` has to be implemented." % lr_name)

    return methodid
#
# def lr_method_id(lr_name, lr_kwargs):
#     """Creates the id for the learning rate method.
#     """
#
#     # listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))
#
#     methodid = lr_method_name_short[lr_name]
#
#     if lr_name == EXPONENTIAL_DECAY:
#         methodid += ".i" + "{:.0e}".format(lr_kwargs["learning_rate"])
#         methodid += ".s" + "{:.0e}".format(lr_kwargs["decay_steps"])
#         methodid += ".r" + "{}".format(lr_kwargs["decay_rate"])
#
#     elif lr_name == POLYNOMIAL_DECAY:
#         methodid += ".i" + "{:.0e}".format(lr_kwargs["learning_rate"])
#         methodid += ".s" + "{:.0e}".format(lr_kwargs["decay_steps"])
#         methodid += ".r" + "{}".format(lr_kwargs["end_learning_rate"])
#         methodid += ".p" + "{}".format(lr_kwargs["power"])
#
#     else:
#         print('----------------------')
#         print('ERROR ', lr_name)
#         raise ValueError("id rule for learning rate `%s` has to be implemented." % lr_name)
#
#     return methodid
