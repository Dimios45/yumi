"""Upload the completed training dataset into a new private Hugging Face repo."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    root=Path('data/lerobot/yumi-block-stack-20261008-training').resolve()
    report=Path('artifacts/hf-upload-training-20261008')
    report.mkdir(exist_ok=True)
    api=HfApi()
    owner=api.whoami()['name']
    if owner!='vruga':
        raise ValueError('Unexpected authenticated account; expected vruga')
    name='yumi-block-stack-20261008-training'
    repo=f'{owner}/{name}'
    suffix=2
    while api.repo_exists(repo,repo_type='dataset'):
        repo=f'{owner}/{name}-v{suffix}'
        suffix+=1
    # Keep the pull command in the dataset card exact if a new suffix was needed.
    card=root/'README.md'
    if repo!=f'{owner}/{name}':
        card.write_text(card.read_text().replace(f'{owner}/{name}',repo))
    files={}
    for p in root.rglob('*'):
        if not p.is_file() or '.cache' in p.relative_to(root).parts:
            continue
        data=p.read_bytes()
        files[str(p.relative_to(root))]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'git_blob_sha1':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()}
    (report/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    api.create_repo(repo_id=repo,repo_type='dataset',private=True,exist_ok=False)
    (report/'created-repo.json').write_text(json.dumps({'repo':repo,'private':True},indent=2)+'\n')
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=root,
        ignore_patterns=['.cache/**'],commit_message='Upload filtered LeRobot v3 training data, source split, and ACT trainer')
    info=api.repo_info(repo,repo_type='dataset',files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    errors=[]
    for name,meta in files.items():
        item=remote.get(name)
        if item is None or item.size!=meta['size']:
            errors.append(f'{name}: missing or size mismatch')
        elif item.lfs and item.lfs.sha256!=meta['sha256']:
            errors.append(f'{name}: LFS hash mismatch')
        elif not item.lfs and item.blob_id!=meta['git_blob_sha1']:
            errors.append(f'{name}: Git blob hash mismatch')
    if errors:
        raise ValueError(errors)
    result={'repo':repo,'url':f'https://huggingface.co/datasets/{repo}','private':info.private,
            'revision':commit.oid,'files_verified':len(files),'bytes':sum(f['size'] for f in files.values()),
            'verification':json.loads((root/'verification.json').read_text())}
    (report/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
