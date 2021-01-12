import trax
from trax.supervised import training
from trax import layers as tl

from model import ReformerLM
from preprocessing.preprocessing import generate_data


def training_loop(n_steps=50, cutoff=0.05, output_dir="./model/"):
    train_gen, eval_gen, vocab_size = generate_data(cutoff)

    lr_schedule = trax.lr.warmup_and_rsqrt_decay(
        n_warmup_steps=1000, max_value=0.01)

    train_task = training.TrainTask(
        # labeled data
        labeled_data=train_gen,
        # loss layer
        loss_layer=tl.CrossEntropyLoss(),
        # optimizer
        optimizer=trax.optimizers.Adam(0.01),
        # lr_schedule
        lr_schedule=lr_schedule,
        # n_steps
        n_steps_per_checkpoint=n_steps
    )

    eval_task = training.EvalTask(
        # labeled data
        labeled_data=eval_gen,
        # metrics
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )

    loop = training.Loop(ReformerLM(vocab_size, 6, mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)

    return loop
