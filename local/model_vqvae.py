import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *
from util import *
from model import *

class quantizer_kotha_arff(nn.Module):
    """
        Input: (B, T, n_channels, vec_len) numeric tensor n_channels == 1 usually
        Output: (B, T, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False, scale=None, assistant=None):
        super().__init__()
        if normalize:
            target_scale = scale if scale is not None else  0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3 #1e-3
            self.normalize_scale = None
        self.embedding0_2classes = nn.Parameter(torch.randn(n_channels, 2, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_3classes = nn.Parameter(torch.randn(n_channels, 3, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_4classes = nn.Parameter(torch.randn(n_channels, 4, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_nclasses = nn.Parameter(torch.randn(n_channels, 16, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()
        self.plot_histogram = 0
        self.assistant = assistant

    def forward(self, x0, chunk_size=512):
        fig = None
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding_2classes = target_norm * self.embedding0_2classes / self.embedding0_2classes.norm(dim=2, keepdim=True)
            embedding_3classes = target_norm * self.embedding0_3classes / self.embedding0_3classes.norm(dim=2, keepdim=True)
            embedding_4classes = target_norm * self.embedding0_4classes / self.embedding0_4classes.norm(dim=2, keepdim=True)
            embedding_nclasses = target_norm * self.embedding0_nclasses / self.embedding0_nclasses.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding_2classes = self.embedding0_2classes
            embedding_3classes = self.embedding0_3classes
            embedding_4classes = self.embedding0_4classes
            embedding_nclasses = self.embedding0_nclasses


        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x1 and embedding: ", x1.shape, embedding.shape)

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks_2classes = []
        index_chunks_3classes = []
        index_chunks_4classes = []
        index_chunks_nclasses = []
        for x1_chunk in x1.split(chunk_size, dim=0):
            #print("Shapes of x1_chunk, embedding_2classes, embedding_3classes and embedding_4classes: ", x1_chunk[:,:,:,:63].shape, embedding_2classes.shape, embedding_3classes.shape, embedding_4classes.shape)
            index_chunks_2classes.append((x1_chunk[:, :,:, 0:64] - embedding_2classes).norm(dim=3).argmin(dim=2))
            index_chunks_3classes.append((x1_chunk[:, :,:,64:128] - embedding_3classes).norm(dim=3).argmin(dim=2))
            index_chunks_4classes.append((x1_chunk[:,:,:,128:192] - embedding_4classes).norm(dim=3).argmin(dim=2))
            index_chunks_nclasses.append((x1_chunk[:,:,:,192:256] - embedding_nclasses).norm(dim=3).argmin(dim=2))

        index_2classes = torch.cat(index_chunks_2classes, dim=0)
        index_3classes = torch.cat(index_chunks_3classes, dim=0)
        index_4classes = torch.cat(index_chunks_4classes, dim=0)
        index_nclasses = torch.cat(index_chunks_nclasses, dim=0)

        # index: (N*samples, n_channels) long tensor
           
        hist_2classes = index_2classes.float().cpu().histc(bins=2, min=-0.5, max=1.5)
        hist_3classes = index_3classes.float().cpu().histc(bins=3, min=-0.5, max=2.5)
        hist_4classes = index_4classes.float().cpu().histc(bins=4, min=-0.5, max=3.5)
        hist_nclasses = index_nclasses.float().cpu().histc(bins=64, min=-0.5, max=3.5)

        if self.plot_histogram:  
               assert self.assistant is not None

               hists = hist_2classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(2) / 2) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_2classes', fig)

               hists = hist_3classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(3) / 3) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_3classes', fig) 
               plt.close()

               hists = hist_4classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(4) / 4) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_4classes', fig) 
               plt.close()

               hists = hist_nclasses.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(64) / 64) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_nclasses', fig) 
               plt.close()

               self.plot_histogram = 0

        prob_2classes = hist_2classes.masked_select(hist_2classes > 0) / len(index_2classes)
        entropy_2classes = - (prob_2classes * prob_2classes.log()).sum().item()

        prob_3classes = hist_3classes.masked_select(hist_3classes > 0) / len(index_3classes)
        entropy_3classes = - (prob_3classes * prob_3classes.log()).sum().item()

        prob_4classes = hist_4classes.masked_select(hist_4classes > 0) / len(index_4classes)
        entropy_4classes = - (prob_4classes * prob_4classes.log()).sum().item()

        prob_nclasses = hist_nclasses.masked_select(hist_nclasses > 0) / len(index_nclasses)
        entropy_nclasses = - (prob_nclasses * prob_nclasses.log()).sum().item()

           
        index1_2classes = (index_2classes + self.offset).view(index_2classes.size(0) * index_2classes.size(1))
        index1_3classes = (index_3classes + self.offset).view(index_3classes.size(0) * index_3classes.size(1))
        index1_4classes = (index_4classes + self.offset).view(index_4classes.size(0) * index_4classes.size(1))
        index1_nclasses = (index_nclasses + self.offset).view(index_nclasses.size(0) * index_nclasses.size(1))

        # index1: (N*samples*n_channels) long tensor
        output_flat_2classes = embedding_2classes.view(-1, embedding_2classes.size(2)).index_select(dim=0, index=index1_2classes)
        output_flat_3classes = embedding_3classes.view(-1, embedding_3classes.size(2)).index_select(dim=0, index=index1_3classes)
        output_flat_4classes = embedding_4classes.view(-1, embedding_4classes.size(2)).index_select(dim=0, index=index1_4classes)
        output_flat_nclasses = embedding_nclasses.view(-1, embedding_nclasses.size(2)).index_select(dim=0, index=index1_nclasses)

        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output_2classes = output_flat_2classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_3classes = output_flat_3classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_4classes = output_flat_4classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_nclasses = output_flat_nclasses.view(x.shape[0], x.shape[1], x.shape[2], -1)

        output = torch.cat([output_2classes, output_3classes, output_4classes, output_nclasses], dim=-1) 
        #print("Shape of output and x: ", output.shape, x.shape, output_2classes.shape)

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses)


    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0_2classes.size(2))
                self.embedding0_2classes.mul_(target_norm / self.embedding0_2classes.norm(dim=2, keepdim=True))

                target_norm = self.embedding_scale * math.sqrt(self.embedding0_3classes.size(2))
                self.embedding0_3classes.mul_(target_norm / self.embedding0_3classes.norm(dim=2, keepdim=True))

                target_norm = self.embedding_scale * math.sqrt(self.embedding0_4classes.size(2))
                self.embedding0_4classes.mul_(target_norm / self.embedding0_4classes.norm(dim=2, keepdim=True))
            
                target_norm = self.embedding_scale * math.sqrt(self.embedding0_nclasses.size(2))
                self.embedding0_nclasses.mul_(target_norm / self.embedding0_nclasses.norm(dim=2, keepdim=True))


    def get_quantizedindices(self, x0, chunk_size=512):
        fig = None
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding_2classes = target_norm * self.embedding0_2classes / self.embedding0_2classes.norm(dim=2, keepdim=True)
            embedding_3classes = target_norm * self.embedding0_3classes / self.embedding0_3classes.norm(dim=2, keepdim=True)
            embedding_4classes = target_norm * self.embedding0_4classes / self.embedding0_4classes.norm(dim=2, keepdim=True)
            embedding_nclasses = target_norm * self.embedding0_nclasses / self.embedding0_nclasses.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding_2classes = self.embedding0_2classes
            embedding_3classes = self.embedding0_3classes
            embedding_4classes = self.embedding0_4classes
            embedding_nclasses = self.embedding0_nclasses


        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x1 and embedding: ", x1.shape, embedding.shape)
            
        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks_2classes = []
        index_chunks_3classes = []
        index_chunks_4classes = []
        index_chunks_nclasses = []
        for x1_chunk in x1.split(chunk_size, dim=0):
            #print("Shapes of x1_chunk, embedding_2classes, embedding_3classes and embedding_4classes: ", x1_chunk[:,:,:,:63].shape, embedding_2classes.shape, embedding_3classes.shape, embedding_4classes$
            index_chunks_2classes.append((x1_chunk[:, :,:, 0:64] - embedding_2classes).norm(dim=3).argmin(dim=2))
            index_chunks_3classes.append((x1_chunk[:, :,:,64:128] - embedding_3classes).norm(dim=3).argmin(dim=2))
            index_chunks_4classes.append((x1_chunk[:,:,:,128:192] - embedding_4classes).norm(dim=3).argmin(dim=2))
            index_chunks_nclasses.append((x1_chunk[:,:,:,192:256] - embedding_nclasses).norm(dim=3).argmin(dim=2))
        
        index_2classes = torch.cat(index_chunks_2classes, dim=0)
        index_3classes = torch.cat(index_chunks_3classes, dim=0)
        index_4classes = torch.cat(index_chunks_4classes, dim=0)
        index_nclasses = torch.cat(index_chunks_nclasses, dim=0)

        # index: (N*samples, n_channels) long tensor

        hist_2classes = index_2classes.float().cpu().histc(bins=2, min=-0.5, max=1.5)
        hist_3classes = index_3classes.float().cpu().histc(bins=3, min=-0.5, max=2.5)
        hist_4classes = index_4classes.float().cpu().histc(bins=4, min=-0.5, max=3.5)
        hist_nclasses = index_nclasses.float().cpu().histc(bins=64, min=-0.5, max=3.5)

        prob_2classes = hist_2classes.masked_select(hist_2classes > 0) / len(index_2classes)
        entropy_2classes = - (prob_2classes * prob_2classes.log()).sum().item()

        prob_3classes = hist_3classes.masked_select(hist_3classes > 0) / len(index_3classes)
        entropy_3classes = - (prob_3classes * prob_3classes.log()).sum().item()
        
        prob_4classes = hist_4classes.masked_select(hist_4classes > 0) / len(index_4classes)
        entropy_4classes = - (prob_4classes * prob_4classes.log()).sum().item()
        
        prob_nclasses = hist_nclasses.masked_select(hist_nclasses > 0) / len(index_nclasses)
        entropy_nclasses = - (prob_nclasses * prob_nclasses.log()).sum().item()

        index1_2classes = (index_2classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_3classes = (index_3classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_4classes = (index_4classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_nclasses = (index_nclasses.squeeze(1) + self.offset).cpu().numpy().tolist()

        latents_2classes = ' '.join(str(k) for k in self.deduplicate(index1_2classes))
        latents_3classes = ' '.join(str(k) for k in self.deduplicate(index1_3classes))
        latents_4classes = ' '.join(str(k) for k in self.deduplicate(index1_4classes))
        latents_nclasses = ' '.join(str(k) for k in self.deduplicate(index1_nclasses))

        print("2 Class Latents and entropy: ", latents_2classes, entropy_2classes)
        print("3 Class Latents and entropy: ", latents_3classes, entropy_3classes)
        print("4 Class Latents and entropy: ", latents_4classes, entropy_4classes)
        print("N Class Latents and entropy: ", latents_nclasses, entropy_nclasses)

    # Remove repeated entries
    def deduplicate(self, arr):
       arr_new = []
       current_element = None
       for element in arr:
          if current_element is None:
            current_element = element
            arr_new.append(element)
          elif element == current_element:
            continue
          else:
            current_element = element
            arr_new.append(element)
       return arr_new


class SILA(nn.Module):     
    def __init__(self, embedding_dim=256, input_dim=80, r = 4, mel_dim = 80, linear_dim = 1025, use_arff = 0, assistant = None):     
        super(SILA, self).__init__()  
        if use_arff:
            self.quantizer = quantizer_kotha_arff(n_channels=1, n_classes=256, vec_len=int(embedding_dim/4), normalize=True, assistant = assistant)     
        else:  
            self.quantizer = quantizer_kotha(n_channels=1, n_classes=16, vec_len=embedding_dim, normalize=True, assistant = assistant)     
        encoder_layers = [     
            (2, 4, 1),    
            (2, 4, 1),     
            (2, 4, 1),     
            (2, 4, 1),     
            (2, 4, 1),    
            (1, 4, 1),
            (2, 4, 1),    
            ]     
        self.downsampling_encoder = DownsamplingEncoderStrict(embedding_dim, encoder_layers, input_dim=mel_dim+128, use_batchnorm=1)     
        #self.decoder = SpecLSTM(input_dim=embedding_dim)  
        self.embedding_fc = nn.Linear(256, 128)  
        #self.decoder.upsample_scales = [2,2]  
        #self.decoder.upsample_network = UpsampleNetwork(self.decoder.upsample_scales)  
        self.r = r 
        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])  
        self.mel_dim = mel_dim
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)
        print("Outputs per step: ", r)
        #self.lid_postnet = CBHG(embedding_dim, K=8, projections=[256, embedding_dim]) 
        self.lid_lstm = nn.LSTM(embedding_dim, 128, bidirectional=True, batch_first=True)
        self.lid_fc = nn.Linear(128, 2)
        self.use_arff = use_arff


    def forward(self, mels, embedding):  
    
        outputs = {}  
        B = mels.size(0)  

        # Add noise to raw audio
        mels_noisy = mels * (0.02 * torch.randn(mels.shape).cuda()).exp() + 0.003 * torch.randn_like(mels)

        #print("Shape of mels: ", mels.shape) 
        mels_downsampled = self.downsampling_encoder(mels_noisy) 
        #print("Shape of mels and mels_downsampled: ", mels.shape, mels_downsampled.shape)
        #mels = mels.view(B, mels.size(1) // self.r, -1) 
        #mels_downsampled = mels_downsampled.view(B, mels_downsampled.size(1) // self.r, -1) 
        #print("Shape of mels and mels_downsampled: ", mels.shape, mels_downsampled.shape)

        # Get approximate phones  
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mels_downsampled.unsqueeze(2))  
        quantized = quantized.squeeze(2)  
 
        # Get the LID logits
        #mels_lid = self.lid_postnet(quantized.transpose(1,2))
        _, (lid_hidden,_) = self.lid_lstm(quantized)
        lid_logits = self.lid_fc(lid_hidden[-1])  

        # Combine inputs  
        emb = embedding.unsqueeze(1).expand(B, mels_downsampled.shape[1], -1)  
        emb = torch.tanh(self.embedding_fc(emb))  
        quantized = torch.cat([quantized, emb], dim=-1) 
    
        # Reconstruction 
        #print("Shapes of quantized and original mels to the deocder: ", quantized.shape, mels.shape)
        mel_outputs, alignments = self.decoder(quantized, mels, memory_lengths=None) 
        #print("Shape of mel outputs: ", mel_outputs.shape)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        #print("Shape of mel outputs: ", mel_outputs.shape)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        #print("Shape of linear outputs: ", linear_outputs.shape)
 
        # Return 
        return mel_outputs, linear_outputs, alignments, lid_logits, vq_penalty.mean(), encoder_penalty.mean(), entropy 


    def forward_getlid(self, mels, embedding):

        B = mels.shape[0]

        emb = embedding.unsqueeze(1).expand(B, mels.shape[1], -1)  
        emb = torch.tanh(self.embedding_fc(emb))  
        mels_noisy = torch.cat([mels, emb], dim=-1) 
        #mels_noisy = mels

        mels_downsampled = self.downsampling_encoder(mels_noisy) 

        # Get approximate phones  
        if self.use_arff:
           quantized, vq_penalty, encoder_penalty, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses = self.quantizer(mels_downsampled.unsqueeze(2))  
        else: 
           latents, entropy = self.quantizer.get_quantizedindices(mels_downsampled.unsqueeze(2))           
           quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mels_downsampled.unsqueeze(2))  

        quantized = quantized.squeeze(2)  
 
        # Combine inputs  
        #emb = embedding.unsqueeze(1).expand(B, mels_downsampled.shape[1], -1)  
        #emb = torch.tanh(self.embedding_fc(emb))  
        #quantized = torch.cat([quantized, emb], dim=-1) 

        # Get the LID logits
        #print("Shape of quantized: ", quantized.shape)
        #quantized = self.lid_postnet(quantized)
        _, (lid_hidden,_) = self.lid_lstm(quantized)
        lid_logits = self.lid_fc(lid_hidden[-1])  

        if self.use_arff:
            return lid_logits, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses
        return lid_logits, entropy

 
    def forward_noreconstruction(self, mels, embedding):  
  
        outputs = {}  
        B = mels.size(0)  
 
        # Add noise to raw audio
        mels_noisy = mels * (0.02 * torch.randn(mels.shape).cuda()).exp() + 0.003 * torch.randn_like(mels)
        #mels_noisy = mels_noisy[:,2::3,:]
 
        emb = embedding.unsqueeze(1).expand(B, mels_noisy.shape[1], -1)  
        emb = torch.tanh(self.embedding_fc(emb))  
        mels_noisy = torch.cat([mels_noisy, emb], dim=-1) 

        #print("Shape of mels: ", mels.shape) 
        mels_downsampled = self.downsampling_encoder(mels_noisy) 
        #print("Shape of mels and mels_downsampled: ", mels.shape, mels_downsampled.shape)
        #mels = mels.view(B, mels.size(1) // self.r, -1) 
        #mels_downsampled = mels_downsampled.view(B, mels_downsampled.size(1) // self.r, -1) 
        #print("Shape of mels and mels_downsampled: ", mels.shape, mels_downsampled.shape)

        # Get approximate phones
        if self.use_arff:
            quantized, vq_penalty, encoder_penalty, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses = self.quantizer(mels_downsampled.unsqueeze(2))  
        else:  
            quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mels_downsampled.unsqueeze(2))  
        quantized = quantized.squeeze(2)  
 
        # Combine inputs  
        #emb = embedding.unsqueeze(1).expand(B, mels_downsampled.shape[1], -1)  
        #emb = torch.tanh(self.embedding_fc(emb))  
        #quantized = torch.cat([quantized, emb], dim=-1) 

        # Get the LID logits
        #print("Shape of quantized: ", quantized.shape)
        #quantized = self.lid_postnet(quantized)
        _, (lid_hidden,_) = self.lid_lstm(quantized)
        lid_logits = self.lid_fc(lid_hidden[-1])  
 
        if self.use_arff:
           return lid_logits, vq_penalty.mean(), encoder_penalty.mean(), entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses  
        return lid_logits, vq_penalty.mean(), encoder_penalty.mean(), entropy 

