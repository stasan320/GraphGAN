void ConfigPath(std::string& path) {
	WCHAR buffer[MAX_PATH];
	GetModuleFileNameW(NULL, buffer, sizeof(buffer) / sizeof(buffer[0]));
	bool t = false;
	int conv;

	for (int i = 0; i < MAX_PATH; i++) {
		if (buffer[wcslen(buffer) - i - 1] != '\\') {
			buffer[wcslen(buffer) - i - 1] = ' ';
		}
		else {
			t = true;
			conv = wcslen(buffer) - 1 - i;
			for (int i = 0; i < conv; i++) {
				path += buffer[i];
				if (buffer[i] == '\\') {
					path += '\\';
				}
			}
			i = MAX_PATH;
		}
	}
}

void ProgramConst(std::string& backup_weight) {
	std::string path;
	ConfigPath(path);
	std::ifstream Config(path + "\\config.txt");

	Config >> backup_weight;
}
