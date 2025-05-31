import os 
def create_symbolic_link(source_path, target_path):
    if not os.path.exists(target_path):
        os.symlink(source_path, target_path)
        print(f'Linked {source_path} to {target_path}')
    else:
        os.remove(target_path)
        os.symlink(source_path, target_path)
        print(f'Relinked {source_path} to {target_path}')