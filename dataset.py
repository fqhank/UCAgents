import random
import json
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset='pathvqa', num_samples=-1, shuffle=False):
        self.dataset = dataset
        self.num_samples = num_samples
        self.shuffle = shuffle

        self.qa_datas = []
        self.cases = []
        self.gt_options = []
        self.temp_list=[]

    def build_all_data(self):
        _ = self.load_data()
        self.build_cases()

        return self.cases, self.gt_options

    def build_cases(self):
        self.cases = []

        l = len(self.qa_datas)
        for i in range(l):
            c = self.qa_datas[i]
            if len(self.temp_list)>0:
                built_case, gt_option = self.create_case(c, self.temp_list[i])
            else:
                built_case, gt_option = self.create_case(c)
            if not built_case:
                continue
            self.cases.append(built_case)
            self.gt_options.append(gt_option)

    def load_data(self):
        self.qa_datas = []

        if self.dataset == 'medqa':
            test_path = f'./data/{self.dataset}/test.jsonl'
            with open(test_path, 'r') as file:
                for line in file:
                    self.qa_datas.append(json.loads(line))

        elif self.dataset == 'pathvqa':
            dataset = load_dataset("flaviagiammarino/path-vqa", split="test")
            for i in range(0,len(dataset)):
                if dataset[i]['answer'] not in ['no', 'yes']:
                    continue
                self.qa_datas.append(dataset[i])

        elif self.dataset == 'vqa-rad':
            dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
            options_path = './data/vqa-rad/answer.json'
            with open(options_path, 'r') as file:
                options = json.load(file)
            for i in range(0,len(dataset)):
                self.qa_datas.append(dataset[i])
                self.temp_list.append(options[i])
            self.qa_datas = [self.qa_datas[k-1] for k in l]

        elif self.dataset == 'slake-vqa':
            dataset = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test")
            options_path = './data/slake-vqa/answer.json'
            with open(options_path, 'r') as file:
                options = json.load(file)
            for i in range(0,len(dataset)):
                if dataset[i]['answer'].lower() not in ['yes','no']:
                    continue
                self.qa_datas.append(dataset[i])
                self.temp_list.append(options[i])
        else:
            print('The dataset is not supported. Exit.')
            exit()

        if self.shuffle:
            random.shuffle(self.qa_datas)
        if self.num_samples != -1 and self.num_samples>0:
            self.qa_datas = self.qa_datas[:self.num_samples]
        print('Test Samples Total: ',len(self.qa_datas))
        return self.qa_datas

    def create_case(self, sample, temp=None):
        if self.dataset == 'medqa':
            question = "Case: " + sample['question'] + "\nOptions:"
            options = []
            for k, v in sample['options'].items():
                options.append("\n({}){}".format(k, v))
            random.shuffle(options)
            question += " ".join(options)
            return {'question': question, 'image': sample['image']}, sample['answer_idx']

        elif self.dataset == 'pathvqa':
            question = 'Question: ' + sample['question'] + '\nOptions:'
            options = [f"(A)yes", f"(B)no"]
            random.shuffle(options)
            question += ' '.join(options)

            pil_image = sample['image'].convert('RGB')
            return {'question': question, 'image': pil_image}, {'yes':'A','no':'B'}[sample['answer']]

        elif self.dataset == 'vqa-rad':
            question = "Case:\n" + sample['question'] + "\nOptions:"
            candi_options = temp['options']
            idx = ['A','B','C','D','E']
            options = []
            answer_idx = None
            for i in range(len(candi_options)):
                options.append("\n({}){}".format(idx[i], candi_options[i]))
                if candi_options[i].lower() == temp['answer'].lower():
                    answer_idx = idx[i]
                
            random.shuffle(options)
            question += " ".join(options)
            pil_image = sample['image'].convert('RGB') 
            return {'question': question, 'image': pil_image}, answer_idx

        elif self.dataset == 'slake-vqa':
            question = "Case: " + sample['question'] + "\nOptions:"
            candi_options = temp['options']
            idx = ['A','B','C','D','E']
            options = []
            answer_idx = None
            for i in range(len(candi_options)):
                options.append("\n({}){}".format(idx[i], candi_options[i]))
                if candi_options[i].lower() == temp['answer'].lower():
                    answer_idx = idx[i]
            random.shuffle(options)
            question += " ".join(options)
            pil_image = sample['image'].convert('RGB') 
            return {'question': question, 'image': pil_image}, answer_idx