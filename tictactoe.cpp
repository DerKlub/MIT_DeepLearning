#include <iostream>
using namespace std;                                                    
char spaces[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};  //initialize array-- including 0 just so that item x and spaces[x] are the same
char player1 = '-', player2 = '-', current;                            //arrays from C, C++ has vectors now  std::vector<variable_type> name = {x, y, z}

char choice;
int winCondition;

void Board(){
    //system("cls");   //cls is the cmd prompt clear terminal screen... "clear" is the linux version of clear terminal
    system("clear");

    std::cout << "Player 1: " << player1 << endl;
    std::cout << "Player 2: " << player2 << endl;
    std::cout << "_____________" << endl;
    std::cout << "|   |   |   |" << endl;
    std::cout << "| " << spaces[7] << " | " << spaces[8] << " | " << spaces[9] << " |" << endl;
    std::cout << "|___|___|___|" << endl;
    std::cout << "|   |   |   |" << endl;
    std::cout << "| " << spaces[4] << " | " << spaces[5] << " | " << spaces[6] << " |" << endl;
    std::cout << "|___|___|___|" << endl;
    std::cout << "|   |   |   |" << endl;
    std::cout << "| " << spaces[1] << " | " << spaces[2] << " | " << spaces[3] << " |" << endl;
    std::cout << "|___|___|___|" << endl;

}


void Input(){
    if (current == player1) {
        std::cout << "\nPlayer 1 Input: ";
    }
    else {
        std::cout << "\nPlayer 2 Input: ";
    }

    while(true) {

        std::cin >> choice;

        if (choice == '1' && spaces[1] == '1') {
            spaces[1] = current;
            break;
        }
        else if (choice == '2' && spaces[2] == '2') {
            spaces[2] = current;
            break;
        }
        else if (choice == '3' && spaces[3] == '3') {
            spaces[3] = current;
            break;
        }
        else if (choice == '4' && spaces[4] == '4') {
            spaces[4] = current;
            break;
        }
        else if (choice == '5' && spaces[5] == '5') {
            spaces[5] = current;
            break;
        }
        else if (choice == '6' && spaces[6] == '6') {
            spaces[6] = current;
            break;
        }
        else if (choice == '7' && spaces[7] == '7') {
            spaces[7] = current;
            break;
        }
        else if (choice == '8' && spaces[8] == '8') {
            spaces[8] = current;
            break;
        }
        else if (choice == '9' && spaces[9] == '9') {
            spaces[9] = current;
            break;
        }
        else {
            std::cout << "Invalid Input >:(" << endl;
        }
    }
}

void TogglePlayer() {
    current = (current == player1)? player2:player1; //if current is player1, switch to player2
}                                                    //if current is not player 1, switch to player 1

int CheckWin() {
    if(spaces[1] == spaces[2] && spaces[2] == spaces[3]){    //horizontal
        return 1;
    }
    else if (spaces[4] == spaces[5] && spaces[5] == spaces[6]){
        return 1;
    }
    else if (spaces[7] == spaces[8] && spaces[8] == spaces[9]){
        return 1;
    }
    else if (spaces[1] == spaces[4] && spaces[4] == spaces[7]){  //vertical
        return 1;
    }
    else if (spaces[2] == spaces[5] && spaces[5] == spaces[8]){
        return 1;
    }
    else if (spaces[3] == spaces[6] && spaces[6] == spaces[9]){
        return 1;
    }
    else if (spaces[1] == spaces[5] && spaces[5] == spaces[9]){ //diagonal
        return 1;
    }
    else if (spaces[3] == spaces[5] && spaces[5] == spaces[7]){
        return 1;
    }
    else if (spaces[1] != '1' && spaces[2] != '2' && spaces[3] != '3' && spaces[4] != '4' && spaces[5] != '5' &&    //draw
            spaces[6] != '6' && spaces[7] != '7' && spaces[8] != '8' && spaces[9] != '9') {
    
        return 2;
    }
    else {
        return 0;
    }

}



int main(void) {
    system("cls");

    std::cout << "\nChoose your symbol\n\n";
    std::cout << "Player 1: ";
    std::cin >> player1;

    player1 = (player1 == 'x' or player1 == 'X')? 'X':'O';  //if player1 is x, player1 is x, if not then player 1 is o
    player2 = (player1 == 'x' or player1 == 'X')? 'O':'X';  //player2 is not player1

    current = player1;

    Board();


    
    do{
        Input();
        Board();
        TogglePlayer();
        winCondition = CheckWin();
    }
    while (winCondition == 0); 

    if(winCondition == 2) {
        std::cout << "DRAW";
    }
    else if(winCondition == 1) {
        if(current == player1){
            std::cout << "Player 2 Wins!";
        }
        else{
            std::cout << "Player 1 Wins!";
        }
    }
    
    

    return 0;
}