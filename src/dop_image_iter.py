class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None,
                 rand_mirror = False, cutoff = 0,
                 ctx_num = 0, images_per_identity = 0,
                 triplet_params = None,
                 mx_model = None,
                 data_name='data', label_name='softmax_label', **kwargs):
           ......
        ## add doppelganger
        self.dop_mode = True
        self.mining = 'dop'
        self.mining_init = -1
        self.rand_ratio = 9/27
        self.dop = np.ones(len(self.seq_identity), dtype=int) * self.mining_init
        
      def doppelganger_reset(self):
        #reset self.oseq by identities seq
        self.dop_cur = 0
        ids = []
        for k in self.id2range:
          ids.append(k)
        random.shuffle(ids)
        self.dop_seq = []
        for _id in ids:
          v = self.id2range[_id]
          _list = range(*v)
          random.shuffle(_list)
          if len(_list)>self.images_per_identity:
            _list = _list[0:self.images_per_identity]
          self.dop_seq += _list
        print('triplet_seq', len(self.dop_seq))
 #      assert len(self.dop_seq)>=self.triplet_bag_size
      def select_doppelganger(self):
        self.seq = []
        while len(self.seq) < self.seq_min_size:
            self.time_reset()
            embeddings = None
            # bag_size = self.triplet_bag_size
            batch_size = self.batch_size
            # data = np.zeros( (bag_size,)+self.data_shape )
            # label = np.zeros( (bag_size,) )
            tag = []
            # idx = np.zeros( (bag_size,) )
            # print('eval %d images..' % bag_size, self.triplet_cur)
            # print('triplet time stat', self.times)
            if self.dop_cur + batch_size > len(self.seq):
                self.doppelganger_reset()
                # bag_size = min(bag_size, len(self.triplet_seq))
                # print('eval %d images..' % bag_size, self.triplet_cur)
            self.times[0] += self.time_elapsed()
            self.time_reset()
            # print(data.shape)
            data = nd.zeros(self.provide_data[0][1])
            label = None
            if self.provide_label is not None:
                label = nd.zeros(self.provide_label[0][1])
            ba = 0
            while True:
                bb = min(ba + batch_size, len(self.seq))
                if ba >= bb:
                    break
                _count = bb - ba
                # data = nd.zeros( (_count,)+self.data_shape )
                # _batch = self.data_iter.next()
                # _data = _batch.data[0].asnumpy()
                # print(_data.shape)
                # _label = _batch.label[0].asnumpy()
                # data[ba:bb,:,:,:] = _data
                # label[ba:bb] = _label
                for i in range(ba, bb):
                    # print(ba, bb, self.triplet_cur, i, len(self.triplet_seq))
                    _idx = self.dop_seq[i + self.dop_cur]
                    s = self.imgrec.read_idx(_idx)
                    header, img = recordio.unpack(s)
                    img = self.imdecode(img)
                    data[i - ba][:] = self.postprocess_data(img)
                    _label = header.label
                    if not isinstance(_label, numbers.Number):
                        _label = _label[0]
                    if label is not None:
                        label[i - ba][:] = _label
                    tag.append((int(_label), _idx))
                    # idx[i] = _idx

                db = mx.io.DataBatch(data=(data,))
                self.mx_model.forward(db, is_train=False)
                net_out = self.mx_model.get_outputs()
                # print('eval for selecting triplets',ba,bb)
                # print(net_out)
                # print(len(net_out))
                # print(net_out[0].asnumpy())
                net_out = net_out[0].asnumpy()
                # print(net_out)
                # print('net_out', net_out.shape)
                if embeddings is None:
                    embeddings = np.zeros((batch_size, net_out.shape[1]))
                embeddings[0:batch_size, :] = net_out
                ba = bb
                # assert len(tag) == bag_size
                self.dop_cur += batch_size
                embeddings = sklearn.preprocessing.normalize(embeddings)
                # self.times[1] += self.time_elapsed()
                # self.time_reset()
                nrof_images_per_class = [1]
                for i in range(1, batch_size):
                    if tag[i][0] == tag[i - 1][0]:
                        nrof_images_per_class[-1] += 1
                    else:
                        nrof_images_per_class.append(1)

                update_dop_cls(embeddings, label,self.dop)  # shape=(T,3)
            # print('found triplets', len(triplets))
            # ba = 0
            # while True:
            #     bb = ba + self.per_batch_size // 3
            #     if bb > len(triplets):
            #         break
            #     _triplets = triplets[ba:bb]
            #     for i in xrange(3):
            #         for triplet in _triplets:
            #             _pos = triplet[i]
            #             _idx = tag[_pos][1]
            #             self.seq.append(_idx)
                ba = bb
                self.times[1] += self.time_elapsed()
 
                 
