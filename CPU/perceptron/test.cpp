	//Conv Neural
  
	
	for (int k = 0; k < 7; k++) {
		for (int l = 0; l < 7; l++) {
			for (int m = 0; m < 4; m++) {
				int SvNum = k * 4 * 28 + 4 * l + m * 28;
				for (int i = SvNum; i < SvNum + 4; i++) {
					std::cout << i << " ";
				}
				std::cout << std::endl;
			}
			/*for (int i = k * 2 * 28 + 4 * l + 28; i < k * 2 * 28 + 4 * l + 4 + 28; i++) {
				std::cout << i << " ";
			}*/
			std::cout << std::endl;
			std::cout << std::endl;
		}


	}


	return 0;
