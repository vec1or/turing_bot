"""Train the causal text model from random weights on attributed source texts."""
import argparse
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from turing_bot.language import DOC, END, PAD, REPLY, Pieces, content


class Block(nn.Module):
    def __init__(self,d,heads):
        super().__init__()
        self.heads=heads
        self.ln1=nn.LayerNorm(d)
        self.ln2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False)
        self.proj=nn.Linear(d,d,bias=False)
        self.up=nn.Linear(d,3*d)
        self.down=nn.Linear(3*d,d)

    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind()
        h=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(h.transpose(1,2).reshape(b,t,d))
        return x+self.down(F.gelu(self.up(self.ln2(x)),approximate="tanh"))


class Net(nn.Module):
    def __init__(self,c):
        super().__init__()
        d=c["dimension"]
        self.embed=nn.Embedding(len(c["vocab"]),d)
        self.pos=nn.Embedding(c["length"],d)
        self.blocks=nn.Sequential(*(Block(d,c["heads"]) for _ in range(c["layers"])))
        self.norm=nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Embedding)):
                nn.init.normal_(m.weight,std=.025)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self,ids):
        x=self.embed(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))
        return F.linear(self.norm(self.blocks(x)),self.embed.weight)


def encode_pairs(rows, tok, width, add_plan=False):
    output=[]
    for r in rows:
        keywords=content(r["response"])
        plan=random.sample(keywords,min(random.randint(1,3),len(keywords))) if add_plan else []
        src=tok.prompt(r["context"],plan)[-73:]
        dest=tok.encode(r["response"])[:width-len(src)-1]+[END]
        xs=src+dest[:-1]
        ys=[-100]*(len(src)-1)+dest
        output.append((xs,ys))
    return output


def batch(rows,indices,device):
    selected=[rows[i] for i in indices]
    length=max(len(x) for x,y in selected)
    xs=np.full((len(selected),length),PAD,dtype=np.int64)
    ys=np.full_like(xs,-100)
    for i,(x,y) in enumerate(selected):
        xs[i,:len(x)]=x
        ys[i,:len(y)]=y
    return torch.from_numpy(xs).to(device),torch.from_numpy(ys).to(device)


def save(net,c,out):
    target=out/"language.tmp.npz"
    np.savez_compressed(target,**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    target.replace(out/"language.npz")
    (out/"language.json").write_text(json.dumps(c,ensure_ascii=False),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("sentences","train","valid","out","report"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--steps",type=int,default=1900)
    p.add_argument("--epochs",type=int,default=7)
    p.add_argument("--batch",type=int,default=48)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    seed=6443
    random.seed(seed)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(3)
    begin=time.monotonic()
    tr,va=[[json.loads(row) for row in f.read_text(encoding="utf-8").splitlines()] for f in (args.train,args.valid)]
    with gzip.open(args.sentences,"rt",encoding="utf-8",newline="") as inp:
        lines=[r["text"] for r in csv.DictReader(inp,delimiter="\t")]
    random.shuffle(lines)
    pieces=Pieces.learn(itertools.chain(lines,(s[k] for s in tr for k in ("response","context"))))
    c=dict(format=1,vocab=pieces.tokens,layers=4,dimension=192,heads=4,length=112)
    cfg={k:v for k,v in c.items() if k!="vocab"}
    tokens=np.asarray([i for line in lines for i in [DOC,*pieces.encode(line),END]],dtype=np.int32)
    encoded,valid=encode_pairs(tr,pieces,c["length"]),encode_pairs(va,pieces,c["length"])
    hinted=encode_pairs(tr,pieces,c["length"],True)
    expansions=[]
    for line in lines[:50000]:
        words=content(line)
        if len(words)<2:
            continue
        plan=random.sample(words,min(random.randint(1,4),len(words)))
        xs=pieces.prompt("",plan)
        dest=pieces.encode(line)[:c["length"]-len(xs)-1]+[END]
        expansions.append((xs+dest[:-1],[-100]*(len(xs)-1)+dest))
    del lines
    net=Net(c).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,betas=(.9,.96),weight_decay=.04)
    autocast=lambda: torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    def loss_at(x,y):
        with autocast():
            logits=net(x)
            return F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
    def update(x,y,lr):
        opt.param_groups[0]["lr"]=lr
        opt.zero_grad(set_to_none=True)
        loss=loss_at(x,y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(),1.)
        scaler.step(opt)
        scaler.update()
        return loss.detach().item()
    report={"seed":seed,"parameters":sum(t.numel() for t in net.parameters()),"tokens_in_monolingual_stream":len(tokens),
            "config":cfg,"train_pairs":len(tr),"validation_pairs":len(va),"steps":args.steps,"batch":args.batch,"pretrain":[],"finetune":[]}
    print("Prepared",report,flush=True)
    def record():
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    average=[]
    block=64
    for step in range(args.steps):
        offsets=rng.integers(0,len(tokens)-block-1,args.batch)
        chunk=torch.tensor(tokens[offsets[:,None]+np.arange(block+1)],dtype=torch.long,device=device)
        lr=.00115*min(1.,(step+1)/60)*(.23+.77*(1+math.cos(math.pi*step/args.steps))/2)
        average.append(update(chunk[:,:-1],chunk[:,1:],lr))
        if (step+1)%100==0:
            report["pretrain"].append([step+1,round(float(np.mean(average)),4),round(time.monotonic()-begin)])
            average=[]
            print("pretrain",report["pretrain"][-1],flush=True)
            record()
    del tokens
    # Real dialogues carry more weight than self-supervised clause-completion pairs.
    indices=np.asarray([i for i,r in enumerate(tr) for _ in range(3 if r["source"] in
                        ("tatoeba_dialogue","everyday","oasst1") else 1)])
    rows=encoded+hinted+expansions
    report["reconstruction_examples"]=len(expansions)
    best=float("inf")
    for epoch in range(args.epochs):
        net.train()
        with_hints=rng.choice(indices,len(indices)//2,replace=False)+len(encoded)
        unsupervised=rng.choice(len(expansions),len(indices)//2,replace=False)+len(encoded)+len(hinted)
        permutation=rng.permutation(np.concatenate((indices,with_hints,unsupervised)))
        # Group similar lengths inside shuffled blocks to avoid wasted GPU memory.
        batches=[]
        for start in range(0,len(permutation),1024):
            block_ids=sorted(permutation[start:start+1024],key=lambda i:len(rows[i][0]))
            batches.extend(block_ids[j:j+args.batch] for j in range(0,len(block_ids),args.batch))
        random.shuffle(batches)
        losses=[update(*batch(rows,ids,device),.00032/(1+.27*epoch)) for ids in batches]
        net.eval()
        total,n=0.,0
        with torch.no_grad():
            for j in range(0,len(valid),args.batch):
                x,y=batch(valid,range(j,min(j+args.batch,len(valid))),device)
                count=int((y!=-100).sum())
                total+=float(loss_at(x,y))*count
                n+=count
        metric=total/n
        report["finetune"].append([epoch+1,round(float(np.mean(losses)),4),round(metric,4),round(time.monotonic()-begin)])
        if metric<best:
            best=metric
            report["best_epoch"]=epoch+1
            save(net,c,args.out)
        print("finetune",report["finetune"][-1],flush=True)
        record()
        if epoch+1-report["best_epoch"]>=2:
            break


if __name__=="__main__":
    main()
