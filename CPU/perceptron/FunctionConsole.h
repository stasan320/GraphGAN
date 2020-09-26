std::string ConfigPath() {
	std::string path;
	WCHAR buffer[MAX_PATH];
	GetModuleFileNameW(NULL, buffer, sizeof(buffer) / sizeof(buffer[0]));

	for (int i = 0; i < MAX_PATH; i++) {
		if (buffer[wcslen(buffer) - i - 1] != '\\') {
			buffer[wcslen(buffer) - i - 1] = ' ';
		}
		else {
			int conv = wcslen(buffer) - 1 - i;
			for (int i = 0; i < conv; i++) {
				path += buffer[i];
				if (buffer[i] == '\\') {
					path += '\\';
				}
			}
			i = MAX_PATH;
		}
	}
	return path;
}

void ProgramConst(std::string* backup) {
	std::ifstream Config(ConfigPath() + "\\config");

	std::vector<std::string> comands;
	comands = { "LastSave", "Step" };

	//while (Config) {
	for(int j = 0; j < comands.size(); j++){
		std::string data, name;
		Config >> data;

		for (int i = 0; i < data.size(); i++) {
			if (data[i] == '=') {
				break;
			}
			name += data[i];
		}
		std::cout << name << " ";

		for (int i = 0; i < comands.size(); i++) {
			if (comands[i] == name) {
				if (i == 0) {
					std::string xz;
					for (int j = 0; j < data.size(); j++) {
						if (data[j] == '=') {
							for (int k = j + 1; k < data.size(); k++) {
								xz += data[k];
							}
							backup[i] = xz;
							break;
						}
					}
					std::cout << backup[i] << std::endl;
				}

				if (i == 1) {
					std::string xz;
					for (int j = 0; j < data.size(); j++) {
						if (data[j] == '=') {
							for (int k = j + 1; k < data.size(); k++) {
								backup[i] += data[k];
							}
							break;
						}
					}
					std::cout << backup[i] << std::endl;
				}
				break;
			}
		}
	}
	//exit(0);
}
