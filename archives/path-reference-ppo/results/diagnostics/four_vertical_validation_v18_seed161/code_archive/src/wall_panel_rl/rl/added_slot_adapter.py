"""Function-preserving curriculum extension for previously absent obstacle slots.

One-static observations use slot 0; slots 1..4 are exactly zero. Freeze the
original actor and adapt only input columns for those previously absent slots.
With fresh zero-moment Adam and no weight decay, one-static actions are preserved.
"""
import torch


ADAPTER_WEIGHTS=('residual.0.weight','prior_gate.0.weight')
CRITIC_PREFIXES=('mlp_extractor.value_net.','value_net.')


def configure_adapter(policy,*,initialize=False,train_critic=False):
    for handle in getattr(policy,'_added_slot_handles',[]):handle.remove()
    parameters=dict(policy.named_parameters());handles=[]
    if not all(name in parameters for name in ADAPTER_WEIGHTS):raise ValueError('Wrong reference-preserving actor')
    for name,p in parameters.items():
        active=name in ADAPTER_WEIGHTS
        p.requires_grad_(active or (train_critic and name.startswith(CRITIC_PREFIXES)))
        if active:
            if p.ndim!=2 or p.shape[1]!=77:raise ValueError('Expected 77-feature first layers')
            if initialize:
                with torch.no_grad():p[:,30:70].zero_()
            mask=torch.zeros_like(p);mask[:,30:70]=1
            handles.append(p.register_hook(lambda gradient,mask=mask:gradient*mask.to(gradient.device)))
    policy._added_slot_handles=handles


def original_actor_preserved(source,candidate):
    a=source.state_dict();b=candidate.state_dict()
    for name,value in a.items():
        if name.startswith(CRITIC_PREFIXES):continue
        if name in ADAPTER_WEIGHTS:
            if not torch.equal(value[:,:30],b[name][:,:30]) or not torch.equal(value[:,70:],b[name][:,70:]):return False
        elif not torch.equal(value,b[name]):return False
    return True
