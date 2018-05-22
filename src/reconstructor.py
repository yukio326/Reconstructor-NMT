#python reconstructor.py [mode (train or dev or test)] [config_path] [best_epoch (only testing)]

import sys
from chainer import *
from utilities import *
from nmt import Encoder, Decoder, AttentionalNMT
import random
import copy

class ReconstructorNMT(Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, resource_vocabulary_size, embed_size, hidden_size, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, use_reconstructor_beamsearch, beam_size, library, pre_nmt):
        super(ReconstructorNMT, self).__init__(
            encoder = Encoder(source_vocabulary_size, embed_size, hidden_size, source_vocabulary, source_word2vec, use_dropout, dropout_rate, library),
            decoder = Decoder(target_vocabulary_size, embed_size, hidden_size, target_vocabulary, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library),
            reconstructor = Decoder(resource_vocabulary_size, embed_size, hidden_size, source_vocabulary, source_word2vec, use_dropout, dropout_rate, generation_limit, use_reconstructor_beamsearch, beam_size, library),
		)
        if pre_nmt is not None:
            copy_model(pre_nmt, self)
        self.resource_vocabulary_size = resource_vocabulary_size
        self.use_beamsearch = use_beamsearch
        self.use_reconstructor_beamsearch = use_reconstructor_beamsearch
        self.generation_limit = generation_limit
        self.library = library

    def __call__(self, batch_source, batch_target):
        self.reset_states()
        encoder_hidden_states = self.encoder(batch_source)
        if batch_target is not None:
            batch_resource = [functions.where(source.data < self.resource_vocabulary_size, source, Variable(self.library.zeros(source.shape, dtype = self.library.int32))) for source in batch_source]
            loss_encoderdecoder, predicts_encoderdecoder, _, _, decoder_hidden_states, _ = self.decoder.forward(encoder_hidden_states, batch_target)
            loss_reconstructor, predicts_reconstructor, _, _, _, _ = self.reconstructor.forward(decoder_hidden_states, batch_resource)
        elif not self.use_reconstructor_beamsearch:
            loss_encoderdecoder, predicts_encoderdecoder = self.decoder(encoder_hidden_states, batch_target)
            loss_reconstructor = None
            predicts_reconstructor = None
        else:
            predicts_encoderdecoder = list()
            initial_beam = [(0, None, list(), encoder_hidden_states, list(), list(), list())]
            decoder_beam = self.decoder.n_forwards(initial_beam)
            new_decoder_beam = list()
            for candidate in decoder_beam:
                logprob, states, sentence, _, embed_states, hidden_states, attention_weights_matrix = candidate
                new_decoder_beam.append((logprob, states, sentence, hidden_states, embed_states, hidden_states, attention_weights_matrix))
            reconstructor_beam = self.reconstructor.n_forwards(new_decoder_beam)
            for _, _, sentence, _, _, _, _ in sorted(reconstructor_beam, key = lambda x: x[0].data / len(x[2]), reverse = True):
                for i, word in enumerate(sentence):
                    if i >= self.generation_limit:
                        break
                    predicts_encoderdecoder.append(word.data)
                    if word.data[0] == 1:
                        break
                break
            loss_encoderdecoder = None
            loss_reconstructor = None
            predicts_reconstructor = None

        return loss_encoderdecoder, loss_reconstructor, predicts_encoderdecoder, predicts_reconstructor

    def reset_states(self):
        self.encoder.reset_states()
        self.decoder.reset_states()
        self.reconstructor.reset_states()

def train(config):
    if len(sys.argv) == 4:
        start = int(sys.argv[3]) - 1
        trace("Start Re-Training ...")
        trace("Loading Vocabulary ...")
        source_vocabulary = Vocabulary.load("{}.{:03d}.source_vocabulary".format(config.model, start))
        target_vocabulary = Vocabulary.load("{}.{:03d}.target_vocabulary".format(config.model, start))
        config.resource_vocabulary_size = min([source_vocabulary.size, config.target_vocabulary_size])
        config.source_vocabulary_size = source_vocabulary.size
        config.target_vocabulary_size = target_vocabulary.size
    else:
        start = 0
        trace("Start Training ...")
        trace("Loading Vocabulary ...")
        source_vocabulary = Vocabulary.load("{}.source_vocabulary".format(config.pre_model))
        target_vocabulary = Vocabulary.load("{}.target_vocabulary".format(config.pre_model))
        config.resource_vocabulary_size = min([source_vocabulary.size, config.target_vocabulary_size])
        config.source_vocabulary_size = source_vocabulary.size
        config.target_vocabulary_size = target_vocabulary.size
    
        if config.use_word2vec == "Load":
            trace("Loading Word2vec ...")
            source_word2vec = load_word2vec(config.source_word2vec_file)
            save_word2vec(source_word2vec, "{}.source_word2vec".format(config.model))
        elif config.use_word2vec == "Make":
            trace("Making Word2vec ...")
            source_word2vec = make_word2vec(config.source_train, config.embed_size)
            save_word2vec(source_word2vec, "{}.source_word2vec".format(config.model))
        else:
            source_word2vec = None

    trace("Making Model ...")
    pre_nmt = AttentionalNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, config.use_dropout, config.dropout_rate, None, False, None, config.library)
    serializers.load_hdf5("{}.weights".format(config.pre_model), pre_nmt)
    
    nmt = ReconstructorNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.resource_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, source_word2vec, None, config.use_dropout, config.dropout_rate, None, False, False, None, config.library, pre_nmt)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()

    opt = config.optimizer
    opt.setup(nmt)
    opt.add_hook(optimizer.GradientClipping(5))

    if start != 0:
        serializers.load_hdf5("{}.{:03d}.weights".format(config.model, start), nmt)
        serializers.load_hdf5("{}.{:03d}.optimizer".format(config.model, start), opt)
        

    for epoch in range(start, config.epoch):
        trace("Epoch {}/{}".format(epoch + 1, config.epoch))
        accum_loss_encoderdecoder = 0.0
        accum_loss_reconstructor = 0.0
        finished = 0
        random.seed(epoch)
        for batch_source, batch_target in random_sorted_parallel_batch(config.source_train, config.target_train, source_vocabulary, target_vocabulary, config.batch_size, config.pooling, config.library):
            nmt.zerograds()
            loss_encoderdecoder, loss_reconstructor, batch_predict_encoderdecoder, batch_predict_reconstructor = nmt(batch_source, batch_target)
            accum_loss_encoderdecoder += loss_encoderdecoder.data
            accum_loss_reconstructor += loss_reconstructor.data
            loss = loss_encoderdecoder + loss_reconstructor
            loss.backward()
            opt.update()

            for source, target, predict_encoderdecoder, predict_reconstructor in zip(convert_wordlist(batch_source, source_vocabulary), convert_wordlist(batch_target, target_vocabulary), convert_wordlist(batch_predict_encoderdecoder, target_vocabulary), convert_wordlist(batch_predict_reconstructor, source_vocabulary)):
                trace("Epoch {}/{}, Sample {}".format(epoch + 1, config.epoch, finished + 1))
                trace("Source         = {}".format(source))
                trace("Target         = {}".format(target))
                trace("Predict_Target = {}".format(predict_encoderdecoder))
                trace("Predict_Source = {}".format(predict_reconstructor))
                finished += 1

        trace("accum_loss_encoderdecoder = {}".format(accum_loss_encoderdecoder))
        trace("accum_loss_reconstructor = {}".format(accum_loss_reconstructor))
        trace("Saving Model ...")
        model = "{}.{:03d}".format(config.model, epoch + 1)
        source_vocabulary.save("{}.source_vocabulary".format(model))
        target_vocabulary.save("{}.target_vocabulary".format(model))
        serializers.save_hdf5("{}.weights".format(model), nmt)
        serializers.save_hdf5("{}.optimizer".format(model), opt)

    trace("Finished.")

def test(config):
    trace("Loading Vocabulary ...")
    source_vocabulary = Vocabulary.load("{}.source_vocabulary".format(config.model))
    target_vocabulary = Vocabulary.load("{}.target_vocabulary".format(config.model))
    config.resource_vocabulary_size = min([source_vocabulary.size, config.target_vocabulary_size])
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    trace("Loading Model ...")
    nmt = ReconstructorNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.resource_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, False, 0.0, config.generation_limit, config.use_beamsearch, config.use_reconstructor_beamsearch, config.beam_size, config.library, None)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()
    serializers.load_hdf5("{}.weights".format(config.model), nmt)

    trace("Generating Translation ...")
    finished = 0
    
    with open(config.predict_file, 'w') as ft:
        for batch_source in mono_batch(config.source_file, source_vocabulary, 1, config.library):
            trace("Sample {} ...".format(finished + 1))
            _, _, batch_predict, _ = nmt(batch_source, None)
            for predict in convert_wordlist(batch_predict, target_vocabulary):
                ft.write("{}\n".format(predict))
                finished += 1

if __name__ == "__main__":
    config = Configuration(sys.argv[1], sys.argv[2])
    if config.mode == "train":
        config.pre_model = "{}.{:03d}".format(config.pre_model, config.pre_best_epoch)
        train(config)
    elif config.mode == "test":
        trace("Start Testing ...")
        config.source_file = config.source_test
        config.predict_file = "{}.test_result.beam{}_RecBeam{}".format(config.model, config.beam_size, config.use_reconstructor_beamsearch)
        config.model = "{}.{:03d}".format(config.model, int(sys.argv[3]))
        test(config)
        trace("Finished.")
    elif config.mode == "dev":
        trace("Start Developing ...")
        config.source_file = config.source_dev
        model = config.model
        if len(sys.argv) == 5:
            start = int(sys.argv[3]) - 1
            end = int(sys.argv[4])
        else:
            start = 0
            end = config.epoch
        for i in range(start, end):
            config.model = "{}.{:03d}".format(model, i + 1)
            trace("Model {}/{}".format(i + 1, config.epoch))
            config.predict_file = "{}.dev_result".format(config.model)
            test(config)
        trace("Finished.")
