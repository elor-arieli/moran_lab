int params[] = {0,1,1000};

int tastedur[] = {100, 1, 100, 1, 1};

int Array1[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

//// array for spout testing, please do not change and use CA bottle for these tests :)
int Array2[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};



int maxITI = 3000;
int up_every_x_licks = 15;
int up_by = 500;

int Water_PIN = 30;
int Nacl_PIN = 31;
int Sugar_PIN = 32;
int CA_PIN = 33;
int Quinone_PIN = 34;

int Nose_Poke1 = 40; // nose poke + IOC
int Nose_Poke2 = 41; // spout

int Laser1 = 20;
int Laser2 = 21;

int Light_1 = 2;

int DigTastes[] = {Water_PIN, Nacl_PIN, Sugar_PIN, CA_PIN, Quinone_PIN};

// basic parameters - do not change
int val1 = HIGH;
int val2 = HIGH;
int i = 0;
int j = 0;
int command = 0;

////////////////////////////////////////////////////////////////////////////////
// set the parameters for the program
int LaserStart = params[0];
int LaserEnd = params[1];
int LaserDuration = LaserEnd - LaserStart;

int TasteDuration = 50; // in milisecs, determine how much drink is given. 
int WaitDuration = params[2];

const int numTrials = 400;

int countTrials = 1;

const int noseOut = 0;

void setup() {
  

  Serial.begin(9600); 
  
  pinMode(Water_PIN,OUTPUT);
  pinMode(Sugar_PIN,OUTPUT);
  pinMode(Nacl_PIN,OUTPUT);
  pinMode(Quinone_PIN,OUTPUT);
  pinMode(CA_PIN,OUTPUT);
  pinMode(Laser1,OUTPUT);
  pinMode(Laser2,OUTPUT);
  //pinMode(IN_PIN,OUTPUT);
  pinMode(Nose_Poke1,INPUT);
  pinMode(Nose_Poke2,INPUT);
  pinMode(Light_1,OUTPUT);

  digitalWrite(Water_PIN, LOW);
  digitalWrite(Sugar_PIN, LOW);
  digitalWrite(Nacl_PIN, LOW);
  digitalWrite(Quinone_PIN, LOW);
  digitalWrite(CA_PIN, LOW);
  digitalWrite(Laser1, LOW);
  digitalWrite(Laser2, LOW);
//  digitalWrite(IN_PIN, LOW);
  digitalWrite(Light_1, HIGH);

  
  
// while (!Serial.available()){}
// digitalWrite(IN_PIN, HIGH);
}

void loop() {

  if (countTrials == numTrials){
    }
  else {
    command = 0;
  
    val1 = digitalRead(Nose_Poke1);   // read the input pin
    val2 = digitalRead(Nose_Poke2);   // read the input pin
  
    if (val1 == LOW) {
      command = Array1[i];
      i++;
    }
    else if (val2 == LOW) {
      command = Array2[j];
      j++;
    }
    if (command == 0){}
    else if (command % 2 == 1){
      Serial.println(command);
      GiveTaste(command);
      delay(WaitDuration-tastedur[command/2]);
      countTrials++;
    }
    else if (command % 2 == 0){
      Serial.println(command);
      if (LaserStart <= tastedur[command/2-1]){
        Taste_Laser_before(command);
      }
      else {
        Taste_Laser_after(command);
      }
      delay(WaitDuration-LaserEnd);
      countTrials++;
    }
  }
}

void GiveTaste(int command){
  int taste = DigTastes[command/2];
  digitalWrite(taste, HIGH);
  delay(tastedur[command/2]);
  digitalWrite(taste, LOW);
  if ((countTrials % up_every_x_licks == 0) && (WaitDuration < maxITI)){
    WaitDuration += up_by;
  }
  if (noseOut == 1){
    waitToGetOut();
  }
}
 
void Taste_Laser_before(int command) {
  int taste = DigTastes[command/2-1];
  digitalWrite(taste, HIGH);
  delay(LaserStart);
  digitalWrite(Laser1, HIGH);
  digitalWrite(Laser2, HIGH);
  delay(tastedur[command/2-1]-LaserStart);
  digitalWrite(taste, LOW);
  delay(LaserDuration-tastedur[command/2-1]);
  digitalWrite(Laser1, LOW);
  digitalWrite(Laser2, LOW);
  if ((countTrials % up_every_x_licks == 0) && (WaitDuration < maxITI)){
    WaitDuration += up_by;
  }
  if (noseOut == 1){
    waitToGetOut();
  }
}

void Taste_Laser_after(int command) {
  int taste = DigTastes[command/2-1];
  digitalWrite(taste, HIGH);
  delay(tastedur[command/2-1]);
  digitalWrite(taste, LOW);
  delay(LaserStart-tastedur[command/2-1]);
  digitalWrite(Laser1, HIGH);
  digitalWrite(Laser2, HIGH);
  delay(LaserDuration);
  digitalWrite(Laser1, LOW);
  digitalWrite(Laser2, LOW);
  if ((countTrials % up_every_x_licks == 0) && (WaitDuration < maxITI)){
    WaitDuration += up_by;
  }
  if (noseOut == 1){
    waitToGetOut();
  }
}

void waitToGetOut(void){
  while ((digitalRead(Nose_Poke1) == LOW) || (digitalRead(Nose_Poke2) == LOW)){}
}

