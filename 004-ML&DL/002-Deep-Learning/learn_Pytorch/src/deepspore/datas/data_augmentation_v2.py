
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#@save_Tensor_with_Augmentation
import torch 
from Bio import SeqIO, SeqRecord
import gzip
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd 
from Pytorch.src.deepspore.utils import run_cmd
from loguru import logger


class Tools:
    '''
    >>> filepath = "data/agumentation/ncbi_genomes_bacillus/ncbi_dataset/data/GCF_045278805.1/GCF_045278805.1_ASM4527880v1_genomic.fna.gz"
    >>> seqs = Tools.parse_fasta(filepath, is_gzip=True)
    >>> seqs_onehot = Tools.fasta_to_onehot(seqs)
    >>> seqs = Tools.onehot_to_seq_fast(seqs_onehot)
    '''

    @staticmethod
    def one_hot(seq: str, device: str = 'cpu') -> torch.Tensor:
        """Vectorized one-hot encoding of a DNA sequence using PyTorch.
        Unknown bases (e.g. N, R, Y) → all-zeros.
        Output: (4, L)
        """
        vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq = seq.upper()
        base_codes = torch.full((len(seq),), fill_value=-1, dtype=torch.long, device=device)
        # Fill valid base positions
        for base, idx in vocab.items():
            mask = torch.tensor([c == base for c in seq], device=device)
            base_codes[mask] = idx
        one_hot = torch.zeros((len(seq), 4), dtype=torch.float32, device=device)
        valid = base_codes >= 0
        one_hot[torch.arange(len(seq), device=device)[valid], base_codes[valid]] = 1.0
        return one_hot.T  # shape: (4, L)
    
    @staticmethod
    def one_hot_fast(seq: str, device: str = 'cpu') -> torch.Tensor:
        """
        Vectorized one-hot encoding of a DNA sequence.
        Unknown bases (e.g. N, R, Y) -> all zeros.
        Output: (4, L)
        """
        seq = seq.upper()
        L = len(seq)
        
        # 构建字符到索引映射表
        # ASCII 'A'->0, 'C'->1, 'G'->2, 'T'->3, 其他->-1
        vocab_map = torch.full((128,), -1, dtype=torch.long, device=device)  # ASCII 0-127
        vocab_map[ord('A')] = 0
        vocab_map[ord('C')] = 1
        vocab_map[ord('G')] = 2
        vocab_map[ord('T')] = 3
        
        # 将序列字符转换为 ASCII 索引
        seq_bytes = torch.tensor([ord(c) for c in seq], dtype=torch.long, device=device)
        
        # 查表得到碱基索引 (-1 表示未知)
        base_idx = vocab_map[seq_bytes]  # shape (L,)
        
        # 创建 one-hot
        onehot = torch.zeros((4, L), dtype=torch.float32, device=device)
        valid = base_idx >= 0
        onehot[base_idx[valid], torch.arange(L, device=device)[valid]] = 1.0
        
        return onehot
    
    @staticmethod
    def onehot_to_seq(seqs_onehot: torch.Tensor):
        """
        Convert one-hot encoded DNA (4,L) to string, handle all-zero columns as 'N'
        """
        vocab = ['A', 'C', 'G', 'T']
        seq_len = seqs_onehot.shape[1]        
        # 最大值索引
        poss = seqs_onehot.argmax(dim=0)
        # 是否全零
        zero_mask = (seqs_onehot.sum(dim=0) == 0)  # shape (L,), True if all zeros
        # 构建序列
        onehot_seqs = []
        for i in range(seq_len):
            if zero_mask[i]:
                onehot_seqs.append('N')  # 全零 -> N
            else:
                onehot_seqs.append(vocab[poss[i]])        
        return ''.join(onehot_seqs)
    
    @staticmethod
    def onehot_to_seq_fast(onehot: torch.Tensor) -> str:
        """
        Vectorized conversion of one-hot (4,L) to DNA sequence string.
        Operating on GPUs or CPUs instead of Python for loop.
        Full-zero columns -> 'N'
        """
        vocab = torch.tensor([ord('A'), ord('C'), ord('G'), ord('T')], dtype=torch.uint8, device=onehot.device)
        # argmax 得到正常碱基索引
        poss = onehot.argmax(dim=0)  # shape (L,)        
        # 是否全零
        zero_mask = (onehot.sum(dim=0) == 0)        
        # 创建一个 uint8 tensor 存字符编码
        seq_uint8 = vocab[poss]           # shape (L,)
        seq_uint8[zero_mask] = ord('N')   # 全零列 -> N        
        # 转成 Python 字符串
        return seq_uint8.cpu().numpy().tobytes().decode('ascii')
    
    @staticmethod
    def parse_fasta(filepath: str, is_gzip: bool = False) -> dict[str, SeqRecord]:
        '''Parse all seqs from gzip compressed format.
        Args:
            filepath: str, the path of fasta file.
            is_gzip: bool, whether the fasta file is gzip compressed.
        Return:
            seqs: dict[str, SeqRecord], the dict of all sequences.
        Demo:
        >>> filepath = "data/agumentation/ncbi_genomes_bacillus/ncbi_dataset/data/GCF_045278805.1/GCF_045278805.1_ASM4527880v1_genomic.fna.gz"
        >>> seqs = Tools.parse_fasta(filepath, is_gzip=True)
        '''
        if is_gzip:
            handle = gzip.open(filepath, 'rt')
        else:
            handle = open(filepath, 'r')
        seqs = SeqIO.parse(handle, format='fasta')
        seqs = SeqIO.to_dict(seqs)
        return seqs
    
    @staticmethod
    def fasta_to_onehot(seqs: dict[str, SeqRecord]) -> dict[str, torch.Tensor]:
        '''Convert all sequences from fasta to one-hot encoding.
        Args:
            seqs: dict[str, SeqRecord], the dict of all sequences.
        Return:
            seqs_onehot: dict[str, torch.Tensor], the dict of all one-hot encoded sequences.
        Demo:
            >>> filepath = "data/agumentation/ncbi_genomes_bacillus/ncbi_dataset/data/GCF_045278805.1/GCF_045278805.1_ASM4527880v1_genomic.fna.gz"
            >>> seqs = Tools.parse_fasta(filepath, is_gzip=True)
            >>> seqs_onehot = Tools.fasta_to_onehot(seqs)
        '''
        seqs_onehot = {id: Tools.one_hot_fast(str(seqrecord.seq), 'cpu') for id, seqrecord in seqs.items()}
        return seqs_onehot


#@save_Tensor_with_Augmentation
class SyntheticData:
    '''
    Example:
    >>> sd = SyntheticData()
    >>> all_df = sd.launch(fold_path = 'data/agumentation/ncbi_genomes_bacillus',\
                            query_fasta_file = 'data/agumentation/Query/sufA.fasta',\
                            result_file = 'cache/result_file.txt')
    >>> SyntheticData.save_df(all_df, 'raw_data.pt', format='torch')
    >>> SyntheticData.save_df(all_df, 'raw_data.csv', format='csv')
    '''
    
    def alignment_with_minimap2(self, genome_fasta_file: str, query_fasta_file: str, result_file: str) -> None:
        '''minimap2 alignment tool.'''
        cmd = f'minimap2 {genome_fasta_file} {query_fasta_file} > {result_file}'
        run_cmd(cmd)

    def parse_minimap2_alignment_results(self, result_file: str) -> str:
        '''Parse the minimap2 results and return label.'''
        found = False
        with open(result_file) as f:
            for line in f:
                fields = line.strip().split("\t")
                qlen = int(fields[1])
                qstart, qend = int(fields[2]), int(fields[3])
                nmatch = int(fields[9])
                alen = int(fields[10])
                identity = nmatch / alen
                coverage = (qend - qstart) / qlen
                if identity >= 0.9 and coverage >= 0.8:
                    found = True
                    break
        return 1 if found else 0

    def alignment_with_mmseqs2(self) -> None:
        '''mmseqs2 alignment tool.'''
        raise NotImplemented
    
    def parse_mmseqs2_alignment_results(self, result_file: str) -> str:
        '''Parse the reuslt aligned from mmseqs2 tool and return label.'''
        raise NotImplemented

    def alignment(self, genome_fasta_file: str, query_fasta_file: str, result_file: str, tool: str) -> str:
        '''Alignment.'''
        logger.debug(f'Start align using {tool}')
        if tool == 'minimap2':
            self.alignment_with_minimap2(genome_fasta_file, query_fasta_file, result_file)
            return result_file
        elif tool == 'mmseqs2':
            self.alignment_with_mmseqs2()
        else:
            raise ValueError(f'Error: Align tool "{tool}" is not exist.')

    def parse_alignment_results(self, result_file: str, tool: str) -> str:
        '''Parse the alignment results.'''
        if tool == 'minimap2':
            label = self.parse_minimap2_alignment_results(result_file)
        elif tool == 'mmseqs2':
            pass
        else:
            raise ValueError(f'Error: There is not exist result of {tool}')
        logger.debug(f"Parse {tool} alignment results, the label is [{label}].")
        return label 
    
    def get_label(self, genome_fasta_file: str, query_fasta_file: str, result_file: str, tool: str = 'minimap2') -> str | int:
        '''Pipeline: Search query sequence to whole genome sequence, return final label for the strain if have match parts.'''
        # step 1: alignment.
        result_file = self.alignment(genome_fasta_file, query_fasta_file, result_file, tool=tool)
        # step 2: parse the alignment results.
        label = self.parse_alignment_results(result_file, tool=tool)
        return label
    
    def to_df(self, seqs: list, seqs_onehot, labels: list) -> pd.DataFrame:
        df_dict = {'Sequences': seqs, 'onehot': seqs_onehot, 'Label': labels}
        df = pd.DataFrame(df_dict)
        logger.debug(f'Construct DataFrame with {df.columns}.')
        return df 
    
    def synthesis(self, genome_fasta_file: str, query_fasta_file: str, result_file: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        '''Synthetic datas.'''
        # step 1: parse and return seqs with dict[str, SeqRecord].
        seqs = Tools.parse_fasta(filepath=genome_fasta_file, is_gzip=(True if genome_fasta_file.endswith('gz') else False))

        # step 2: get the label.
        label = self.get_label(genome_fasta_file, query_fasta_file, result_file)

        # step 3: onehot and return with dict[str, torch.Tensor], then save to local disk.
        seqs_onehot = Tools.fasta_to_onehot(seqs)
        genome_onehot_file = f'{genome_fasta_file}.pt'
        torch.save(seqs_onehot, genome_onehot_file)

        # step 3: constructe the df.
        df = self.to_df([genome_fasta_file], [genome_onehot_file], [label])
        
        return df 
    
    def batch_synthesis(self, all_genome_fasta_files: list, query_fasta_file: str, result_file: str) -> pd.DataFrame:
        '''Batch synthesis for all genome fasta files.'''
        all_df = pd.DataFrame(columns=['Sequences', 'onehot', 'Label'])
        for genome_fasta_file in tqdm(all_genome_fasta_files, desc='Strain', position=0, leave=True):
            strain_df = self.synthesis(genome_fasta_file, query_fasta_file, result_file)
            all_df = pd.concat([all_df, strain_df], axis=0, ignore_index=True)
        return all_df
    
    def find_all_genome_fasta_file(self, fold_path: str) -> list[str]:
        '''Find all genome fasta files in the fold path.'''
        fold = Path(fold_path)
        files = list(fold.rglob("*.fna.gz"))
        # files = list(fold.rglob("*.fna"))
        # file_paths = [str(file.resolve()) for file in files if file.name.endswith('_genomic.fna.gz')]
        file_paths = [str(file.resolve()) for file in files if "cds_from_genomic" not in file.name]
        return file_paths
    
    def lanuch(self, fold_path: str, query_fasta_file: str, result_file: str) -> pd.DataFrame:
        '''Lanuch the batch synthesis.
        >>> all_df = sd.lanuch(fold_path = 'data/agumentation/ncbi_genomes_bacillus',\
                                query_fasta_file = 'data/agumentation/Query/sufA.fasta',\
                                result_file = 'cache/result_file.txt') 
        '''
        all_genome_fasta_files = self.find_all_genome_fasta_file(fold_path)
        # all_df= self.batch_synthesis(all_genome_fasta_files[0:10], query_fasta_file, result_file)
        all_df= self.batch_synthesis(all_genome_fasta_files, query_fasta_file, result_file)
        return all_df
    
    @staticmethod
    def save_df(df: pd.DataFrame, raw_data_file: str, format: str = 'torch') -> None:
        if format == 'csv':
            df.to_csv(raw_data_file, index=False)   
        elif format == 'torch':
            torch.save(df, raw_data_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_df(raw_data_file: str, format: str = 'torch') -> pd.DataFrame:
        if format == 'csv':
            df = pd.read_csv(raw_data_file)
        elif format == 'torch':
            df = torch.load(raw_data_file, weights_only=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return df
    

# # Test
# sd = SyntheticData()
# all_df = sd.lanuch(fold_path = 'data/agumentation/ncbi_genomes_bacillus', query_fasta_file = 'data/agumentation/Query/sufA.fasta', result_file = 'cache/result_file.txt')
# SyntheticData.save_df(all_df, './data/agumentation/raw_data.pt', format='torch')

# sd = SyntheticData()
# all_df = sd.lanuch(fold_path = 'data/agumentation/ncbi_genomes_bacillus', query_fasta_file = 'data/agumentation/Query/sufA.fasta', result_file = 'cache/result_file.txt')
# SyntheticData.save_df(all_df, './data/agumentation/raw_data.csv', format='csv')


#@save_Tensor_with_Augmentation
from Bio import SeqRecord
import random


class DNASequenceAugmentation:
    '''Augmentate genomic sequences in DNA sequence level.
    >>> sd = DNASequenceAugmentation()
    >>> seqs = sd.reversed_complement(seqs)
    >>> seqs = sd.random_linearization(seqs)
    >>> seqs = sd.random_order_chro_plasmid(seqs)
    >>> seqs = sd.concate_seqs_to_genome(seqs)
    >>> genome = sd.augmentation(seqs)
    '''

    def reversed_complement(self, seqs: dict[str, SeqRecord], p: float = 0.5) -> dict[str, SeqRecord]:
        '''Reverse comlement DNA with 50% probability per sequence in all sequences'''
        if isinstance(seqs, dict):
            new_seqs = {}
            for id, seq in seqs.items():
                if random.random() < p:
                    seq.seq = seq.seq.reverse_complement()
                new_seqs[id] = seq
            return new_seqs
    
    def random_linearization(self, seqs: dict[str, SeqRecord], linear_position: int = None) -> dict[str, SeqRecord]:
        '''Randomly linearlize the cycle chromsome or plasmid.'''
        if isinstance(seqs, dict):
            for id, seq in seqs.items():
                if linear_position is not None:
                    pos = random.randint(0, linear_position)
                else:
                    pos = random.randint(0, len(seq.seq)-1)
                seq.seq = seq.seq[pos:] + seq.seq[:pos]
        return seqs
    
    def random_concatenation(self, seqs: dict[str, SeqRecord], shuffle: bool = False) -> dict[str, SeqRecord]:
        '''Shuffle the seqs for adding noise, avoiding model to memorize the order between chromsome and plasmids.'''
        if isinstance(seqs, dict):
            seq_ids = list(seqs.keys())
            if shuffle:
                random.shuffle(seq_ids)
        new_seqs = {id: seqs.get(id) for id in seq_ids}
        return new_seqs
    
    def concate_seqs_to_genome(self, seqs: dict[str, SeqRecord]) -> str:
        '''Concate all sequences to one genome sequence.'''
        genome = [str(seq.seq) for _, seq in seqs.items()]
        genome = ''.join(genome)
        return genome
    
    def augmentation(self, seqs: dict[str, SeqRecord]) -> str:
        '''Agumentation for all sequences of genome.'''
        # dict[str, SeqRecord] -> dict[str, SeqRecord]
        seqs = self.reversed_complement(seqs)

        # dict[str, SeqRecord] -> dict[str, SeqRecord]
        seqs = self.random_linearization(seqs)

        # dict[str, SeqRecord] -> dict[str, SeqRecord]
        seqs = self.random_concatenation(seqs)

        # dict[str, SeqRecord] -> str
        genome = self.concate_seqs_to_genome(seqs)
        
        return genome


#@save_Tensor_with_Augmentation
import torch
# import random


class DNAAugmentation:
    '''DNA sequence augmentation on one-hot encoded genome (4, L)
    Include:
      - Revers complement
      - Random linearization
      - Random concatenation
      - Random masking, mutation, deletion, insertion

    Example:
    >>> dna_augmentation = DNAAugmentation()
    >>> seqs_onehot = dna_augmentation.reversed_complement(seqs_onehot)
    >>> seqs_onehot = dna_augmentation.random_linearization(seqs_onehot)
    >>> seqs_onehot = dna_augmentation.random_concatenation(seqs_onehot) # onehot sequences with dict[str, Tensor]: (4, L)
    >>> genome = dna_augmentation.augmentation(seqs_onehot) # onehot sequences with Tensor: (4, L)
    '''
    
    # -------------------------------------- #
    # ----- Augmentation on genome level --- #
    # -------------------------------------- #
    def reversed_complement(self, seqs_onehot: dict[str, torch.Tensor], p: float = 0.5) -> dict[str, torch.Tensor]:
        """Reverse complement for one-hot encoded DNA sequences. Each sequence shape = (4, L), channels = A,C,G,T"""
        rc_seqs = {}
        # ACGT:0123 rc-> TGCA:3210
        idx = torch.tensor([3, 2, 1, 0], device=next(iter(seqs_onehot.values())).device)  # A<->T, C<->G
        for id, seq in seqs_onehot.items():
            if seq.ndim != 2 or seq.shape[0] != 4:
                raise ValueError(f"Expected shape (4, L), got {seq.shape}")
            # if random.random() < p:
            if torch.rand(1).item() < p:    # generate rand number between 0 and 1 by torch.rand()
                x = torch.flip(seq, dims=[1])                   # reverse
                x = x.index_select(dim= 0, index= idx)          # complement
                rc_seqs[id] = x
            else:
                rc_seqs[id] = seq
        return rc_seqs

    def random_linearization(self, seqs_onehot: dict[str, torch.Tensor], linear_position: int = None) -> dict[str, torch.Tensor]:
        """Random linearization of circular DNA sequences. Equivalent to rolling the sequence along length dimension (dim=1)."""
        new_seqs_onehot = {}
        for id, seq in seqs_onehot.items():
            if linear_position is not None:
                # shift = random.randint(0, linear_position)
                shift = torch.randint(0, linear_position + 1, (1,)).item() # shift with torch.randint
            else:
                # shift = random.randint(0, seq.shape[1] - 1)  # shift randomly along L
                shift = torch.randint(0, seq.shape[1], (1,)).item() # with torch.randint

            new_seqs_onehot[id] = torch.roll(seq, shifts=shift, dims=1)
        return new_seqs_onehot

    def random_concatenation(self, seqs_onehot: dict[str, torch.Tensor], shuffle: bool = False) -> dict[str, torch.Tensor]:
        """Random concatenation of multiple sequences (chromosomes/plasmids). Output: single long sequence, shape = (4, L_total)"""
        ids = list(seqs_onehot.keys())
        if shuffle:
            # random.shuffle(ids)  # shuffle order
            perm = torch.randperm(len(ids)) # with torch.randperm 不放回
            ids = [ids[i] for i in perm]
        new_seqs = {id: seqs_onehot[id] for id in ids}
        return new_seqs
    
    def concate_seqs_to_genome(self, seqs: dict[str, torch.Tensor]) -> torch.Tensor:
        '''Concate all sequences to one genome sequence.'''
        genome = [seq for _, seq in seqs.items()]
        genome = torch.cat(genome, dim=1)
        return genome

    # -------------------------------------- #
    # --- Augmentation on sequence level  -- #
    # -------------------------------------- #

    def random_mask(self, genome: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
        '''Mask random regions (set to 0).'''
        L = genome.size(1)
        num_mask = int(L * ratio) # required mask numbers
        if num_mask == 0:
            return genome
        else:
            idx = torch.randperm(L)[:num_mask]
            genome[:, idx] = 0
            return genome
        
    def random_mutation(self, genome: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
        """Vectorized random mutation for one-hot DNA genome.
        params:
            - genome: Tensor (4, L), where rows = [A,C,G,T]
            - ratio: fraction of positions to mutate
        """
        assert genome.ndim == 2 and genome.shape[0] == 4, f"Expected shape (4, L), got {genome.shape}"
        L = genome.shape[1]
        num_mut = int(L * ratio)
        if num_mut == 0:
            return genome

        device = genome.device

        # --- 1. 随机选取突变位置 ---
        idx = torch.randperm(L, device=device)[:num_mut]  # (num_mut,)

        # --- 2. 找出每个位置的当前碱基索引 ---
        # argmax over channels (A,C,G,T)
        old_base = torch.argmax(genome[:, idx], dim=0)  # (num_mut,)

        # --- 3. 随机生成新碱基 ---
        new_base = torch.randint(0, 4, (num_mut,), device=device)

        # --- 4. 确保新碱基 != 原碱基 ---
        # 当新碱基与旧碱基相同时，重新采样
        conflict = new_base == old_base
        while conflict.any():
            new_base[conflict] = torch.randint(0, 4, (conflict.sum(),), device=device)
            conflict = new_base == old_base

        # --- 5. 将旧碱基位置清零 ---
        genome[:, idx] = 0

        # --- 6. 按新碱基索引置 1 ---
        genome[new_base, idx] = 1

        return genome

        
    def random_deletion(self, genome: torch.Tensor, ratio: float = 0.02) -> torch.Tensor:
        """Randomly delete some positions."""
        L = genome.shape[1]
        num_del = int(L * ratio)
        if num_del == 0: 
            return genome
        else:
            idx = torch.randperm(L)[:num_del]
            keep_mask = torch.ones(L, dtype=torch.bool, device=genome.device)
            keep_mask[idx] = False
            return genome[:, keep_mask]
        
    def random_insertion(self, genome: torch.Tensor, ratio: float = 0.02) -> torch.Tensor:
        """Vectorized random insertion for one-hot encoded DNA genome.
        params:
            - genome: Tensor (4, L)
            - ratio: fraction of positions to insert new bases
        """
        assert genome.ndim == 2 and genome.shape[0] == 4, f"Expected shape (4, L), got {genome.shape}"
        L = genome.shape[1]
        num_ins = int(L * ratio)
        if num_ins == 0:
            return genome

        device = genome.device

        # --- 1. 生成递增插入位置 ---
        idx = torch.sort(torch.randperm(L, device=device)[:num_ins])[0]

        # --- 2. 创建插入掩码 ---
        expanded = torch.zeros(L + num_ins, dtype=torch.bool, device=device)
        expanded[idx + torch.arange(num_ins, device=device)] = True

        # --- 3. 随机插入碱基 ---
        new_bases = torch.zeros(4, num_ins, device=device)
        new_base_idx = torch.randint(0, 4, (num_ins,), device=device)
        new_bases[new_base_idx, torch.arange(num_ins, device=device)] = 1  # one-hot bases

        # --- 4. 构造结果 ---
        result = torch.zeros(4, L + num_ins, device=device)
        result[:, expanded] = new_bases
        result[:, ~expanded] = genome

        return result

    def augmentation(self, seqs_onehot: dict[str, torch.Tensor]) -> torch.Tensor:
        """Full augmentation pipeline."""
        # dict[str, Tensor] -> dict[str, Tensor]
        seqs_onehot = self.reversed_complement(seqs_onehot, p=0.5)

        # dict[str, Tensor] -> dict[str, Tensor]
        seqs_onehot = self.random_linearization(seqs_onehot)

        # dict[str, Tensor] -> dict[str, Tensor]
        seqs_onehot = self.random_concatenation(seqs_onehot, shuffle=True)

        # dict[str, Tensor] -> Tensor
        genome = self.concate_seqs_to_genome(seqs_onehot)

        # independent augmentations (each optional)
        # Tensor -> Tensor
        # genome = self.random_mask(genome, ratio=0.05)

        # # Tensor -> Tensor
        # genome = self.random_mutation(genome, ratio=0.05)

        # # Tensor -> Tensor
        # genome = self.random_deletion(genome, ratio=0.02)
        
        # # Tensor -> Tensor
        # genome = self.random_insertion(genome, ratio=0.02)
        
        return genome


#@save_Tensor_with_Augmentation
import torch 
import pandas as pd 


class GenomicDataset(torch.utils.data.Dataset):
    ''' 
    Example:
    >>> df = SyntheticData.load_data('raw_data.parquet')
    >>> all_datasets = GenommicDataset(df=df, is_augmentation=True)
    '''
    
    def __init__(self, df: pd.DataFrame, is_augmentation: bool) -> None:
        super().__init__()
        self.df = self.load_to_memory(df=df.copy()) # copy
        self.dna_augmentation = DNAAugmentation()
        self.is_augmentation = is_augmentation

    def load_to_memory(self, df: pd.DataFrame):
        '''replace file path with Tensor by torch.load funchont.'''
        df['onehot'] = df['onehot'].map(lambda x: torch.load(x, weights_only=True))
        return df

    def __getitem__(self, idx):
        '''Lazily lanuch, get augmentation per epoch.'''
        seqs_onehot = self.df.iloc[idx]['onehot']
        label = self.df.iloc[idx]['Label']
        if self.is_augmentation:
            augmented_seq = self.dna_augmentation.augmentation(seqs_onehot)  # Tensor (4, L)
        else:
            augmented_seq = self.dna_augmentation.concate_seqs_to_genome(seqs_onehot) # Tensor (4, L)
        return augmented_seq, label
    
    def __len__(self):
        return len(self.df)


#@save_Tensor_with_Augmentation
import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt


class GenomicDataLoader:
    '''
    Example:
    >>> loader = GenomicDataLoader().get_dataloader(
            all_dataset=all_dataset, 
            batch_size=64, 
            num_workers=8, 
            pin_memory=True, 
            cut_off=10000, 
            use_balanced_sampler=True
        )
    '''

    def collate_fn(self, batch, cut_off: int = 10_000_000) -> dict[str, torch.Tensor | list]:
        """Pads a list of (input_tensor, label) pairs to the same sequence length."""
        inputs, labels = zip(*batch)

        lengths = [x.shape[1] for x in inputs]  # each x is (4, L)
        max_len = min(max(lengths), cut_off)    # respect cut_off

        batch_size = len(inputs)
        padded = torch.zeros((batch_size, 4, max_len), dtype=torch.float32)   # (B, 4, L_max)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)           # (B, L_max)

        for i, x in enumerate(inputs):
            seq_len = min(x.shape[1], max_len)
            padded[i, :, :seq_len] = x[:, :seq_len]   # left-align + cut_off
            mask[i, :seq_len] = 1                     # valid tokens

        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return {'inputs': padded, 'labels': labels, 'attention_mask': mask}

    def get_labels(self, dataset: torch.utils.data.Dataset):
        """Extract labels from dataset."""
        return [dataset[i][1] for i in range(len(dataset))]

    def get_sampler(self, dataset: torch.utils.data.Dataset, labels: list | None = None):
        """Create WeightedRandomSampler to handle imbalanced data."""
        if labels is None:
            labels = self.get_labels(dataset)

        label_counts = Counter(labels)
        weights = [1.0 / label_counts[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True
        )
        return sampler

    def get_dataloader(self,
                       all_dataset: torch.utils.data.Dataset,
                       batch_size: int,
                       num_workers: int,
                       pin_memory: bool,
                       cut_off: int = 10_000_000,
                       use_balanced_sampler: bool = True) -> DataLoader:
        """Return DataLoader with optional WeightedRandomSampler for imbalance handling."""

        sampler = None
        shuffle = False
        if use_balanced_sampler:
            sampler = self.get_sampler(all_dataset)
        else:
            shuffle = True  # only enable shuffle when not using sampler

        loader = DataLoader(
            dataset=all_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: self.collate_fn(x, cut_off=cut_off),
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=sampler
        )
        return loader

    # -------------------------------
    # 📊 新增: 类别分布可视化
    # -------------------------------
    def plot_label_distribution(self, dataset: torch.utils.data.Dataset, title="Label Distribution"):
        '''Visulization of label distribution for all Dataset.'''

        labels = self.get_labels(dataset)
        counter = Counter(labels)

        # plt.figure(figsize=(6, 4))
        plt.bar(counter.keys(), counter.values())
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title(title)
        plt.show()

    def plot_sampled_distribution(self, dataloader: DataLoader, num_batches: int = 100, title="Sampled Distribution"):
        """采样若干 batch 后统计类别分布，验证 WeightedRandomSampler 是否平衡"""
        
        counts = Counter()
        seen = 0

        for i, batch in enumerate(dataloader):
            labels = batch['labels'].tolist()
            counts.update(labels)
            seen += len(labels)
            if i >= num_batches:
                break

        # plt.figure(figsize=(6, 4))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel("Class Label")
        plt.ylabel("Sampled Count")
        plt.title(f"{title} (n={seen})")
        plt.show()