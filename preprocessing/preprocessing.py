from .raw import read_raw
import random
import trax


def generate_data(cut_off=0.05):
    # 形式为  title: 拂霓裳 author: 晏殊 text: xxxxxxx
    r = read_raw()
    data_set = ["题：%s 作：%s 文：%s" % (i["title"], i["author"], i["text"]) for i in r]
    cut_off = int(len(data_set) * cut_off)
    train_data, eval_data = data_set[:-cut_off], data_set[-cut_off:]

    # vocab化（留两个空位）
    vocabs = ['', ''] + list(set("".join(data_set)))
    vocab_size = len(vocabs)
    vocab_file = "".join(["'%s'\n" % i for i in vocabs])
    with open('generated/vocab/songci.subword', "w+") as f:
        f.write(vocab_file)
        f.close()

    def stream(data):
        while True:
            d = random.choice(data)
            yield (d, d)

    data_pipeline = trax.data.Serial(
        trax.data.Shuffle(),
        trax.data.Tokenize(vocab_dir="generated/vocab", vocab_file="songci.subword"),
        trax.data.FilterByLength(2048),
        trax.data.BucketByLength(boundaries=[128, 256,  512, 1024],
                                 batch_sizes=[16,    8,    4,   2, 1]),
        trax.data.AddLossWeights(id_to_mask=0)
    )

    return data_pipeline(stream(train_data)), data_pipeline(stream(eval_data)), vocab_size
