import logging

logNN = logging.getLogger('NN')
logNN.propagate = False
logNN.setLevel(logging.DEBUG)

fh1 = logging.FileHandler('log/mod_sel.log')
fh1.setLevel(logging.DEBUG)

ch1 = logging.StreamHandler()
ch1.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh1.setFormatter(formatter)
ch1.setFormatter(formatter)

logNN.addHandler(fh1)
logNN.addHandler(ch1)
