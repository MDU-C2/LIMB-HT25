#include <stdio.h>

#define WINDOW_SIZE
#define OVERLAP
#define BUFFER_SIZE (WINDOW_SIZE + OVERLAP)

int samples[BUFFER_SIZE];
int sample_index = 0;
int total_samples = 0;

void main()[int value == 0;

            while (1) value =  // Read from ADC-channel
            samples[sample_index] = value;
            sample_index = (sample_index + 1) % BUFFER_SIZE; total_samples++;

            if (total_samples >= WINDOW_SIZE &&
                (total_samples - WINDOW_SIZE) % OVERLAP == 0){}

]