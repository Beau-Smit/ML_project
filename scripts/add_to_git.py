from pathlib import Path
import subprocess
from itertools import islice
import subprocess


def main():
	precip_path = Path.cwd() / '..' / 'data' / 'precip'
	n = 250
	all_paths = [path for path in precip_path.iterdir()]
	file_chunks = [all_paths[i * n:(i + 1) * n] for i in range((len(all_paths) + n - 1) // n )] 
	chunk_num = 47
	for file_chunk in file_chunks:
		command = ['git', 'add']
		command.extend(file_chunk)
		subprocess.run(command)
		subprocess.run(['git', 'commit', '-m', 'addition '+str(chunk_num)])
		subprocess.run(['git', 'push', 'origin', 'data:master'])
		chunk_num += 1

if __name__ == '__main__':
	main()