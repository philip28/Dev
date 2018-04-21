import tensorflow as tf
from tensorflow.python.saved_model.signature_constants\
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY

NUM_SAMPLES = 1000

def learning_rate_fn(base_lr, global_step, batch_size, num_classes):
    base_lr = tf.cast(base_lr, tf.float32)
    global_step = tf.cast(global_step, tf.float32)

    epoch_size = (num_classes-1) * NUM_SAMPLES * 6 * 2
    steps_per_epoch = epoch_size / batch_size
    steps_per_epoch = tf.cast(steps_per_epoch, tf.float32)

    gs = tf.cast(tf.cast(global_step/steps_per_epoch, tf.int32), tf.float32)
    lr = base_lr / tf.sqrt(gs + 1)

    return lr


######################################################################3
def model_alexnet_v2(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    input_layer = features['image']

    # 1 convolutional layer ************************************************************************
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[11,11],
        strides=[4,4],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    conv1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")


    # 2 convolutional layer ************************************************************************
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=192,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    conv2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 3 convolutional layer ************************************************************************
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")

    # 4 convolutional layer ************************************************************************
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4")

    # 5 convolutional layer ************************************************************************
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5")

    conv5 = tf.layers.max_pooling2d(inputs=conv5,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 6 convolutional layer (dense replacement)******************************************************
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=4096,
        kernel_size=[6, 6],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    conv6 = tf.layers.dropout(inputs=conv6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 7 convolutional layer (dense replacement)******************************************************
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=4096,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv7")

    conv7 = tf.layers.dropout(inputs=conv7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=conv7,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def model_alexnet_r(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    input_layer = features['image']

    # 1 convolutional layer ************************************************************************
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11,11],
        strides=[4,4],
        #kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
        #bias_initializer=tf.initializers.zeros(),
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    conv1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")


    # 2 convolutional layer ************************************************************************
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=256,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    conv2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 3 convolutional layer ************************************************************************
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")

    # 4 convolutional layer ************************************************************************
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4")

    # 5 convolutional layer ************************************************************************
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5")

    conv5 = tf.layers.max_pooling2d(inputs=conv5,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 6 convolutional layer (dense replacement)******************************************************
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=4096,
        kernel_size=[6, 6],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    conv6 = tf.layers.dropout(inputs=conv6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 7 convolutional layer (dense replacement)******************************************************
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=4096,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv7")

    conv7 = tf.layers.dropout(inputs=conv7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=conv7,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def model_alexnet_light(features, labels, mode, params):
    # size 227
    # input layer

    #tf.identity(labels, name='labels')
    #tf.identity(features['file'], name='files')

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    input_layer = features['image']

    # 1 convolutional layer ************************************************************************
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=48,
        kernel_size=[11,11],
        strides=[4,4],
        #kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
        #bias_initializer=tf.initializers.zeros(),
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    # It seems that these kinds of layers have a minimal impact and are not used any more.
    # Basically, their role have been outplayed by other regularization techniques (such as dropout and batch normalization),
    # better initializations and training methods. This is what is written in the lecture notes for the Stanford Course CS321n on ConvNets
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    conv1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")


    # 2 convolutional layer ************************************************************************
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=128,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    conv2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 3 convolutional layer ************************************************************************
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=192,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")

    # 4 convolutional layer ************************************************************************
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=192,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4")

    # 5 convolutional layer ************************************************************************
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5")

    conv5 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 6 convolutional layer (dense replacement)******************************************************
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=2048,
        kernel_size=[6, 6],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    conv6 = tf.layers.dropout(inputs=conv6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 7 convolutional layer (dense replacement)******************************************************
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=2048,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv7")

    conv7 = tf.layers.dropout(inputs=conv7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=conv7,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    #if not loss_filter_fn:
    #    def loss_filter_fn(name):
    #        return 'batch_normalization' not in name

    ## Add weight decay to the loss.
    #loss = cross_entropy + weight_decay * tf.add_n(
    #    [tf.nn.l2_loss(v) for v in tf.trainable_variables()
    #    if loss_filter_fn(v.name)])
    loss = cross_entropy


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


######################################################################3
def alexnet_rs(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
       # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    net = features['image']

    net = tf.layers.conv2d(
        inputs=net,
        filters=96,
        kernel_size=[5,5],
        strides=[4,4],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=256,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    ## 3 convolutional layer ************************************************************************
    #conv3 = tf.layers.conv2d(
    #    inputs=conv2,
    #    filters=384,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv3")

    ## 4 convolutional layer ************************************************************************
    #conv4 = tf.layers.conv2d(
    #    inputs=conv3,
    #    filters=384,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv4")

    ## 5 convolutional layer ************************************************************************
    #conv5 = tf.layers.conv2d(
    #    inputs=conv4,
    #    filters=256,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv5")
    
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[6,6],
        strides=[1,1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[1,1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc7")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=net,
        filters=params["num_classes"],
        kernel_size=[1,1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    class_weights = tf.constant(params['class_weights'])
    weighted_logits = tf.multiply(logits, class_weights)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=weighted_logits, onehot_labels=labels)

    #cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(params['learning_rate'], global_step, params['batch_size'], params['num_classes'])

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def model_alexnet_s(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    net = features['image']

    net = tf.layers.conv2d(
        inputs=net,
        filters=96,
        kernel_size=[11,11],
        strides=[4,4],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=256,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    ## 3 convolutional layer ************************************************************************
    #conv3 = tf.layers.conv2d(
    #    inputs=conv2,
    #    filters=384,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv3")

    ## 4 convolutional layer ************************************************************************
    #conv4 = tf.layers.conv2d(
    #    inputs=conv3,
    #    filters=384,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv4")

    ## 5 convolutional layer ************************************************************************
    #conv5 = tf.layers.conv2d(
    #    inputs=conv4,
    #    filters=256,
    #    kernel_size=[3,3],
    #    strides=[1,1],
    #    padding="same",
    #    activation=tf.nn.relu,
    #    name="conv5")
    
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[6,6],
        strides=[1,1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[1,1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc7")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=net,
        filters=params["num_classes"],
        kernel_size=[1,1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    class_weights = tf.constant(params['class_weights'])
    weighted_logits = tf.multiply(logits, class_weights)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=weighted_logits, onehot_labels=labels)

    #cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(params['learning_rate'], global_step, params['batch_size'], params['num_classes'])

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def vgg11_r(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    net = features['image']

    net = tf.layers.conv2d(
        inputs=net,
        filters=64,
        kernel_size=[3,3],
        strides=[2,2],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    net = tf.layers.conv2d(
        inputs=net,
        filters=128,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[2, 2],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=192,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3_1")
    net = tf.layers.conv2d(
        inputs=net,
        filters=192,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3_2")
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[2, 2],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_1")
    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_2")
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[2, 2],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_1")
    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_2")
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[2, 2],
                                    strides=[2,2],
                                    padding="valid")


    net = tf.layers.conv2d(
        inputs=net,
        filters=2048,
        kernel_size=[7, 7],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc6")
    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.conv2d(
        inputs=net,
        filters=2048,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc7")
    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=net,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    class_weights = tf.constant(params['class_weights'])
    weighted_logits = tf.multiply(logits, class_weights)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=weighted_logits, onehot_labels=labels)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    #cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(params['learning_rate'], global_step, params['batch_size'], params['num_classes'])

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def fcn6(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    net = features['image']

    net = tf.layers.conv2d(
        inputs=net,
        filters=96,
        kernel_size=[3,3],
        strides=[2,2],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    net = tf.layers.conv2d(
        inputs=net,
        filters=96,
        kernel_size=[3,3],
        strides=[2,2],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")


    net = tf.layers.conv2d(
        inputs=net,
        filters=256,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")

    net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3,3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4")

    net = tf.layers.conv2d(
        inputs=net,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5")

    net = tf.layers.conv2d(
        inputs=net,
        filters=256,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv6")
    
    net = tf.layers.max_pooling2d(inputs=net,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[6,6],
        strides=[1,1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc7")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.conv2d(
        inputs=net,
        filters=4096,
        kernel_size=[1,1],
        strides=[1,1],
        padding="valid",
        activation=tf.nn.relu,
        name="fc8")

    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=net,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    class_weights = tf.constant(params['class_weights'])
    weighted_logits = tf.multiply(logits, class_weights)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=weighted_logits, onehot_labels=labels)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    #cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(params['learning_rate'], global_step, params['batch_size'], params['num_classes'])

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

######################################################################3
def model_alexnet(features, labels, mode, params):
    # size 227
    # input layer

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Generate a summary node for the images
        tf.summary.image('images', features['image'], max_outputs=6)

    input_layer = features['image']

    # 1 convolutional layer ************************************************************************
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11,11],
        strides=[4,4],
        #kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
        #bias_initializer=tf.initializers.zeros(),
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    # It seems that these kinds of layers have a minimal impact and are not used any more.
    # Basically, their role have been outplayed by other regularization techniques (such as dropout and batch normalization),
    # better initializations and training methods. This is what is written in the lecture notes for the Stanford Course CS321n on ConvNets
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    conv1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")


    # 2 convolutional layer ************************************************************************
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=256,
        kernel_size=[5,5],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    conv2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 3 convolutional layer ************************************************************************
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")

    # 4 convolutional layer ************************************************************************
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv4")

    # 5 convolutional layer ************************************************************************
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3,3],
        strides=[1,1],
        padding="same",
        activation=tf.nn.relu,
        name="conv5")
    
    conv5 = tf.layers.max_pooling2d(inputs=conv5,
                                    pool_size=[3, 3],
                                    strides=[2,2],
                                    padding="valid")

    # 6 convolutional layer (dense replacement)******************************************************
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=4096,
        kernel_size=[6, 6],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv6")

    conv6 = tf.layers.dropout(inputs=conv6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 7 convolutional layer (dense replacement)******************************************************
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=4096,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu,
        name="conv7")

    conv7 = tf.layers.dropout(inputs=conv7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits conv with filters = classes
    conv_scores = tf.layers.conv2d(
        inputs=conv7,
        filters=params["num_classes"],
        kernel_size=[1, 1],
        padding="valid",
        activation=None,
        name="conv_scores")

    logits = tf.reshape(conv_scores,
                        [-1, params["num_classes"]],
                        name="logits")

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "files": features['file'],
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        export_outputs = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"probabilities": tf.nn.softmax(conv_scores, name="softmax_tensor")})
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    class_weights = tf.constant(params['class_weights'])
    weighted_logits = tf.multiply(logits, class_weights)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=weighted_logits, onehot_labels=labels)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    #cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(params['learning_rate'], global_step, params['batch_size'], params['num_classes'])

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update ops to be added as a dependency to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)