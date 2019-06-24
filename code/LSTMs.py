class LSTMs(Model):
    """LSTMs model from the CVPR paper
       Link: http://openaccess.thecvf.com/content_cvpr_2016/papers/Ma_Learning_Activity_Progression_CVPR_2016_paper.pdf
    """
    def __init__(self,
                 config,
                 input_dim=1,
                 output_dim=2,
                 cell_type="LSTM",
                 lam=0.0,
                 chop_prop=1.0):
        self.NAME = "LSTMs"
        super(LSTMs, self).__init__(config=config)

        # --- hyperparameters ---
        self.CHOP_PROP = chop_prop # THIS DECIDES WHAT PROPORTION OF THE TIME SERIES TO USE - LEAVE AT 1.0
        self.HIDDEN_DIM = config["model"]["hidden_dim"]
        self.LAMBDA = lam

        # --- mappings ---
        self.rnn = nn.LSTM(input_dim, self.HIDDEN_DIM)
        self.out = nn.Linear(self.HIDDEN_DIM, output_dim)

        # --- Non-linearities ---
        self.softmax = nn.Softmax(dim=2)

    def forward(self, sequence):
        self.hidden = self.initHidden() # REPLACE WITH YOUR OWN CODE
        sequence = sequence[:int(self.CHOP_PROP*len(sequence))]
        output_seq, hidden_seq = self.rnn(sequence, self.hidden)
        output = self.out(output_seq)
        self.logit_seq = self.softmax(output)
        logits = self.logit_seq[-1]
        return logits

    def applyLoss(self, logits, labels): #, training_mode):
        """Core difference between standard LSTM and the LSTM-s"""
        # --- Classification Loss ---
        criterion = self.defineLoss(self._LOSS_NAME) # REPLACE THIS WITH YOUR OWN CODE - Cross Entropy is fine
        loss_c = criterion(logits, labels)

        # --- Detection Score Loss ---
        new_logit_seq = []
        for i in range(self.logit_seq.shape[0]):
            if i == 0:
                new_logit_seq.append(self.logit_seq[i] - self.logit_seq[i])
            if i > 0:
                prev_max, _ = self.logit_seq[:i].max(0)
                new_logit_seq.append(prev_max - self.logit_seq[i])
        new_logit_seq = torch.stack(new_logit_seq, 0).clamp(0)
        loss_s = torch.sum(new_logit_seq)

        # --- compute final loss ---
        loss = loss_c + self.LAMBDA*loss_s
        return loss
