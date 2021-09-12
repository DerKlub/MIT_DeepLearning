#include <iostream>
#include <vector>
#include <string>

int main(void) {
  std::string input;
  std::vector<char> vowels = {'a', 'e', 'i', 'o', 'u'};
  std::vector<char> result;



  std::cout << "Input a word or phrase: ";
  std::cin >> input;

  for(int i = 0; i < input.size(); i++) {
    for(int j = 0; j < vowels.size(); j++) {
      if(input[i] == vowels[j]) {
        result.push_back(input[i]);
      }
    }
  }
  for(int k = 0; k < result.size(); k++) {
    std::cout << result[k];
  }
  return 0;
}