import argparse
import time
from tqdm import tqdm
import torch.utils.data
from datasets import *
from utils import *

from models.danet import Changer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nochange_list = ["the scene is the same as before", 
                 "there is no difference",
                 "the two scenes seem identical", 
                 "no change has occurred",
                 "almost nothing has changed"]


def save_captions(args, vocab, gt_captions, pred_captions):
    result_json_file = {}
    reference_json_file = {}

    for i, item in enumerate(pred_captions):
        line_hypo = []
        for word_idx in item:
            word = get_key(vocab, word_idx)
            line_hypo.append(word[0])
        result_json_file[str(i)] = ' '.join(line_hypo)

    for i, item in tqdm(enumerate(gt_captions)):
        reference_json_file[str(i)] = []
        for sentence in item:
            line_repo = []
            for word_idx in sentence:
                word = get_key(vocab, word_idx)
                line_repo.append(word[0])
            reference_json_file[str(i)].append(' '.join(line_repo))


    with open('eval_results_fortest/' + args.split + '/' + args.exp_name + '_res.json', 'w') as f:
        json.dump([result_json_file, pred_captions], f)

    with open('eval_results_fortest/' + args.split + '/' + args.exp_name + '_gts.json', 'w') as f:
        json.dump([reference_json_file, gt_captions], f)


def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]


def evaluate_transformer(args, device, model, vocab, dataloader, save_label=False):
    beam_size = args.beam_size

    vocab_inverse = {v:k for k,v in vocab.items()}
    
    gt_captions = []
    pred_captions = []
    gt_labels = []
    pred_labels = []
    raw_pred_captions = []
    
    gt_change_captions = []
    gt_no_change_captions = []
    pred_change_captions = []
    pred_no_change_captions = []

    with torch.inference_mode():
        
        for i, (image_pairs, allcaps) in enumerate(tqdm(dataloader, desc=args.split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # image_pairs [b, 2, 3, w, h]
            # allcaps [b, 5, 52]
            
            image_pairs = image_pairs.to(device)  # [1, 2, 3, 256, 256]
            image1 = image_pairs[:, 0, :, :, :]
            image2 = image_pairs[:, 1, :, :, :]

            outputs, _ = model.decode_beam(image1, image2, beam_size, return_all=False)
 
            for ii, sent in enumerate(outputs.tolist()):

                gt_caption = list(map(lambda c: [w for w in c if w not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]], allcaps[ii].tolist()))  
                gt_captions.append(gt_caption)
    
                pred_caption = [w for w in sent if w not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
                pred_captions.append(pred_caption)

                assert len(gt_captions) == len(pred_captions)

                raw_gt_caption = []
                for word in gt_caption[0]:
                    raw_gt_caption.append(vocab_inverse[word])
                raw_gt_caption = " ".join(raw_gt_caption)
                
                raw_pred_caption = []
                for word in pred_caption:
                    raw_pred_caption.append(vocab_inverse[word])
                raw_pred_caption = " ".join(raw_pred_caption)
 

                if raw_pred_caption in nochange_list:
                    pred_label = 0
                else:
                    pred_label = 1

                if raw_gt_caption in nochange_list: 
                    gt_label = 0
                    gt_no_change_captions.append(gt_caption)
                    pred_no_change_captions.append(pred_caption)
                else: 
                    gt_label = 1
                    gt_change_captions.append(gt_caption)                
                    pred_change_captions.append(pred_caption)

                gt_labels.append(gt_label)
                pred_labels.append(pred_label)
                raw_pred_captions.append(raw_pred_caption)


    print(sum(pred_labels))

    save_captions(args, vocab, gt_captions, pred_captions)
    json.dump(raw_pred_captions, open('result.json', 'w'))

    results = compute_precision_recall(gt_labels, pred_labels)
    for cls, metrics in results.items():
        print(f"Class {cls}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")    


    # print('nochange_metric:')
    # nochange_metric = get_eval_score(gt_no_change_captions, pred_no_change_captions)
    # print(nochange_metric)

    # print('change_metric:')
    # change_metric = get_eval_score(gt_change_captions, pred_change_captions)
    # print(change_metric)

    print(".......................................................")
    metrics = get_eval_score(gt_captions, pred_captions)
    
    scores = (metrics["Bleu_4"] + metrics["METEOR"] + metrics["ROUGE_L"] + metrics["CIDEr"]) / 4 * 100

    print("beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {} s* {}".format
            (args.beam_size, 
             round(metrics["Bleu_1"] * 100, 2), 
             round(metrics["Bleu_2"] * 100, 2), 
             round(metrics["Bleu_3"] * 100, 2), 
             round(metrics["Bleu_4"] * 100, 2), 
             round(metrics["METEOR"] * 100, 2), 
             round(metrics["ROUGE_L"] * 100, 2), 
             round(metrics["CIDEr"] * 100, 2),
             round(scores, 2)
        ))

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change_Captioning')
    parser.add_argument('--data_folder', type=str, default="./data/levir-cc/")
    parser.add_argument('--data_name', type=str, default="LEVIR_CC_5_cap_per_img_5_min_word_freq")    
    parser.add_argument('--exp_dir', type=str, default='exps')
    parser.add_argument('--exp_name', type=str, required=True)    

    parser.add_argument('--n_enc', type=int, required=True)
    parser.add_argument('--n_dec', type=int, required=True)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--enc_dropout', type=float, default=0.)
    parser.add_argument('--dec_dropout', type=float, default=0.)

    parser.add_argument('--split', default="TEST")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--beam_size', type=int, default=3)

    parser.add_argument('--save_label', action='store_true')


    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.exp_dir, args.exp_name)):
        raise Exception(f"Path does not exist! {os.path.join(args.exp_dir, args.exp_name)}")

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    vocan_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(vocan_file, 'r') as j:
        vocab = json.load(j)
    args.vocab = vocab

    model = Changer(args)
    model.set_finetune(False)

    names = ['backbone', 'encoder', 'decoder']
    print_component_params(model, names)  

    checkpoint_path = os.path.join(args.exp_dir, args.exp_name, 'BEST_checkpoint.pth.tar')
    ckpt = torch.load(checkpoint_path, map_location=str(device))['model']
    msg = model.load_state_dict(ckpt, strict=True)
    print(msg)

    model = model.to(device)

    dataloader = torch.utils.data.DataLoader(
        CaptionDataset(args, 'TEST'), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True)    
    
    model.eval()
  
    metrics = evaluate_transformer(args, device, model, vocab, dataloader, save_label=args.save_label)    
