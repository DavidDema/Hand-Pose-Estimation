/* Copyright (C) 2012-2017 Ultraleap Limited. All rights reserved.
 *
 * Use of this code is subject to the terms of the Ultraleap SDK agreement
 * available at https://central.leapmotion.com/agreements/SdkAgreement unless
 * Ultraleap has signed a separate license agreement with you or your
 * organisation.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <time.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "LeapC.h"
#include "ExampleConnection.h"

#define PORT 8080
#define TEST false

int64_t lastFrameID = 0; //The last frame received

int main(int argc, char** argv) {
  // ------------------------------
  // Leap setup
  // ------------------------------
  printf("LeapSocketServer V1.0\n");
  OpenConnection();
  while(!IsConnected)
    millisleep(100); //wait a bit to let the connection complete

  printf("Connected Leap.");
  LEAP_DEVICE_INFO* deviceProps = GetDeviceProperties();
  if(deviceProps)
    printf("Using device %s.\n", deviceProps->serial);

  // ------------------------------
  // Socket setup
  // ------------------------------

  int sockfd, newsockfd;
  socklen_t clilen;
  char buffer[1024];
  struct sockaddr_in serv_addr, cli_addr;
  int n;

  printf("Creating socket.");
  // create socket
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
      perror("ERROR opening socket");
      exit(1);
  }

  printf("Binding socket.\n");
  // bind socket to address
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(PORT);
  if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
      perror("ERROR on binding");
      exit(1);
  }

  printf("Listening for incoming socket connections.");
  // listen for incoming connections
  listen(sockfd, 5);
  clilen = sizeof(cli_addr);

  //printf("Accepting.\n");
  // accept incoming connection
  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
  if (newsockfd < 0) {
      perror("ERROR on accept");
      exit(1);
  }

  printf("Start sending frames.");
  for(;;){
    // Start time
    clock_t start_time = clock(); 
    
    LEAP_TRACKING_EVENT *frame = GetFrame();
    if(frame && (frame->tracking_frame_id > lastFrameID)){
      lastFrameID = frame->tracking_frame_id;
      //printf("Frame %lli with %i hands.\n", (long long int)frame->tracking_frame_id, frame->nHands);

      bool status = false;
      float data_send[19];

      if(frame->nHands<1){
        float data[] = {
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        };
        memcpy(data_send, data, sizeof(data));
      }else{
        //for(uint32_t h = 0; h < frame->nHands; h++){
        LEAP_HAND* hand = &frame->pHands[0];
        
        if(TEST==false){
          float data[] = {
            hand->palm.position.x,
            hand->palm.position.y,
            hand->palm.position.z,
            hand->thumb.distal.next_joint.x,
            hand->thumb.distal.next_joint.y,
            hand->thumb.distal.next_joint.z,
            hand->index.distal.next_joint.x,
            hand->index.distal.next_joint.y,
            hand->index.distal.next_joint.z,
            hand->middle.distal.next_joint.x,
            hand->middle.distal.next_joint.y,
            hand->middle.distal.next_joint.z,
            hand->ring.distal.next_joint.x,
            hand->ring.distal.next_joint.y,
            hand->ring.distal.next_joint.z,
            hand->pinky.distal.next_joint.x,
            hand->pinky.distal.next_joint.y,
            hand->pinky.distal.next_joint.z,
            hand->pinch_distance
          };
          memcpy(data_send, data, sizeof(data));
        }else{
          float data[] = {
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0
          };
          memcpy(data_send, data, sizeof(data));
        }
        status = true;
      }
      char buffer_data[sizeof(data_send)];

      // End time
      clock_t end_time = clock();

      // Calculate time 
      double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
      double fps = 1/time_taken;

      if(status==true)
      {
        printf("sending valid frame\n");
        //printf("fps:%.2f\n", fps);
        memcpy(buffer_data, data_send, sizeof(data_send));
        n = send(newsockfd, buffer_data, sizeof(buffer_data), 0);
        if (n < 0) {
            perror("ERROR writing data to socket");
            exit(1);
        }
      }else{
        printf("invalid frame, not sending !\n");
        //printf("fps:%.2f (invalid frame sent)\n", fps);
      }
    }
    //millisleep(100);
  } //ctrl-c to exit
  close(newsockfd);
  close(sockfd);
  return 0;
}
//End-of-Sample
