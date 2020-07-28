import ctranslate2

def load_matrix():
    fn = "/nfs/project/users/xingshi/vph/tmp/prefix_info.txt"
    ms = []
    buffers = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line == "":
                ms.append(list(buffers))
                buffers = []
            else:
                ll = [int(x) for x in line.split()]
                buffers.append(ll)
    return ms

def main():
    t = ctranslate2.Translator('/nfs/project/users/xingshi/vph/data/smart_reply/run/v1.trainb_200_bpe_20k.k3.base/export_ct2','cuda')
    with open('/nfs/project/users/xingshi/vph/tmp/source.txt', encoding = 'utf8') as f:
        line = f.readline()
    r = t.translate_batch([line.split()], beam_size = 1)
    print('With translate_batch')
    print(r[0][0]['tokens'])

    ms = load_matrix()
    em = ms[0]
    tm = ms[1]
    lm = ms[2]
    init_state = ms[3][0][0]
    r = t.translate_batch_with_fsa_prefix([line.split()], em, tm, lm, init_state, beam_size = 1)
    print('With translate_batch_with_fsa_prefix')
    print(r[0][0]['tokens'])

    print(ms)
    em = []
    tm = []
    lm = []
    init_state = -2
    r = t.translate_batch_with_fsa_prefix([line.split()], em, tm, lm, init_state, beam_size = 1)
    print('With translate_batch_with_fsa_prefix, -2')
    print(r[0][0]['tokens'])

main()