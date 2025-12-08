//+------------------------------------------------------------------+
//|                                                    lstm_test.mq5 |
//|                                                            Auron |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Auron"
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| External parameters for script operation                         |
//+------------------------------------------------------------------+
// Name of the file with the training sample
input string   StudyFileName  = "study_data.csv";
// File name for recording the error dynamics
input string   OutputFileName = "loss_study_lstm.csv";
input string   AccuracyFileName = "accuracy_study_lstm.csv";
// Number of historical bars in one pattern
input int BarsToLine = 40;
// Number of input layer neurons per 1 bar
input int NeuronsToBar = 4;
// Use OpenCL
input bool UseOpenCL = true;
// Packet size for updating the weights matrix          
input int BatchSize = 10000;
// Learning rate
input double LearningRate = 3e-5;
// Number of hidden neurons in LSTM layer
input int HiddenLayer  = 128;
// Number of Hidden layers after the LSTM layer
input int HiddenLayers = 0; 
// Number of iterations of updating the weights matrix
input int Epochs  = 500;
// Note: LSTM composition layer works on CPU; GPU support will be added later
// Note: Disable UseOpenCL = false for LSTM to work properly with current implementation

//+------------------------------------------------------------------+
//| Connect the neural network library                               |
//+------------------------------------------------------------------+
#include "..\realization\neuronnet.mqh"

//+------------------------------------------------------------------+
//| Early Stopping Class                                             |
//+------------------------------------------------------------------+
class CEarlyStopping
{
private:
   int               m_patience;
   double            m_min_delta;
   int               m_wait;
   double            m_best_loss;
   bool              m_stopped;

public:
                     CEarlyStopping(int patience = 5, double min_delta = 0.0)
                     : m_patience(patience), m_min_delta(min_delta), m_wait(0), m_best_loss(DBL_MAX), m_stopped(false) {}
                    ~CEarlyStopping() {}

   bool              Check(double current_loss)
     {
      if(current_loss < m_best_loss - m_min_delta)
        {
         m_best_loss = current_loss;
         m_wait = 0;
        }
      else
        {
         m_wait++;
         if(m_wait >= m_patience)
           {
            m_stopped = true;
            return true;
           }
        }
      return false;
     }

   bool              IsStopped() const { return m_stopped; }
   void              Reset() { m_wait = 0; m_best_loss = DBL_MAX; m_stopped = false; }
};
//+------------------------------------------------------------------+
//| Beginning of the script program                                  |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("Starting LSTM Test...");
   //--- prepare a vector to store the network error history
   VECTOR loss_history = VECTOR::Zeros(Epochs);
   VECTOR accuracy_history = VECTOR::Zeros(Epochs);
   
   //--- 1. Initialize model
   CNet net;
   if(!NetworkInitialize(net)) {
      Print("NetworkInitialize failed!");
      return;
   }
   Print("Network Initialized.");
      
   //--- 2. Load the training sample data
   CArrayObj data;
   CArrayObj targets;
   if(!LoadTrainingData(StudyFileName, data, targets)) {
      Print("LoadTrainingData failed!");
      return;
   }
   Print("Training Data Loaded. Total samples: ", data.Total());
      
   //--- 3. Train model 
   if(!NetworkFit(net, data, targets, loss_history, accuracy_history)) {
      Print("NetworkFit failed!");
      return;
   }
   Print("Network Fit Complete.");
      
   //--- 4. Save the error history of the model
   SaveLossHistory(OutputFileName, loss_history);
   SaveLossHistory(AccuracyFileName, accuracy_history);
   
   //--- 5. Save the obtained model
   net.Save("StudyLSTM.net");
   Print("Done");
}

//+------------------------------------------------------------------+
//| Initializing the model                                           |
//+------------------------------------------------------------------+
bool NetworkInitialize(CNet &net)
  {
   CArrayObj layers;
   //--- create a description of the network layers
   if(!CreateLayersDesc(layers))
      return false;
   //--- initialize the network
   if(!net.Create(&layers, (TYPE)LearningRate, (TYPE)0.9, (TYPE)0.999, LOSS_MSE, 0, 0))
     {
      PrintFormat("Error of init Net: %d", GetLastError());
      return false;
     }
   
   net.UseOpenCL(UseOpenCL);
   // Note: UseOpenCL already initializes OpenCL and registers buffers.
   // Calling InitOpenCL again causes double registration which fails on BufferWrite.
   
   // Use a smaller smoothing factor for better loss visibility during training
   // (typical range: 10-100, not batch size)
   net.LossSmoothFactor(50);
   return true;
  }

//+------------------------------------------------------------------+
//| Create layer descriptions                                        |
//+------------------------------------------------------------------+
bool CreateLayersDesc(CArrayObj &layers)
  {
   CLayerDescription *descr;
   
   //--- 1. Input Layer (Base)
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type         = defNeuronBase;
   descr.count        = NeuronsToBar * BarsToLine;
   descr.window       = 0;
   descr.activation   = AF_NONE;
   descr.optimization = None;
   if(!layers.Add(descr))
      return false;
      
   int prev_count = descr.count;
   
   
   //--- 1.1 Batch Normalization Layer
   
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type         = defNeuronBatchNorm;
   descr.count        = prev_count;
   descr.window       = prev_count;
   descr.batch        = BatchSize;
   descr.activation   = AF_NONE;
   descr.optimization = Adam;
   if(!layers.Add(descr))
      return false;
   
   
   //--- 2. LSTM Layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type         = defNeuronLSTM;
   descr.count        = HiddenLayer;
   descr.window       = prev_count; // Input size for LSTM
   descr.batch        = BarsToLine; // Set BPTT depth
   descr.activation   = AF_NONE;
   descr.optimization = Adam;
   if(!layers.Add(descr))
      return false;
      
   prev_count = descr.count;
   
   /*
   //--- 2.1 Dropout Layer
   
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type         = defNeuronDropout;
   descr.count        = prev_count;
   descr.window       = prev_count;
   descr.probability  = 0.2; // 20% dropout
   descr.activation   = AF_SWISH;
   descr.optimization = None;
   if(!layers.Add(descr))
      return false;
   
   
   /*   
   //--- 3. Hidden Layers
   for(int i = 1; i <= HiddenLayers; i++){
      if(!(descr = new CLayerDescription()))
         return false; 
      descr.type        = defNeuronBase;
      descr.count       = HiddenLayer;
      descr.window      = prev_count;
      descr.activation  = AF_SWISH;
      descr.optimization = Adam;
      if(!layers.Add(descr))
         return false;
   
      prev_count = descr.count; 
   }
   return true; 
   */
    
   //--- 4. Output Layer (Dense)
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type         = defNeuronBase;
   descr.count        = 2; // Buy, Sell
   descr.window       = prev_count;
   descr.activation   = AF_TANH;
   descr.optimization = Adam;
   if(!layers.Add(descr))
      return false;
   
      
   return true;
  }
//+------------------------------------------------------------------+
//| Load training data                                               |
//+------------------------------------------------------------------+
bool LoadTrainingData(string file_name, CArrayObj &data, CArrayObj &targets)
  {
   Print("Loading training data from: ", file_name);
   int handle = FileOpen(file_name, FILE_READ | FILE_CSV | FILE_ANSI, ",");
   if(handle == INVALID_HANDLE) {
      Print("Failed to open file: ", file_name);
      return false;
   }
   
   //--- display the progress of training data loading in the chart comment
   uint next_comment_time = 0;
   enum
     {
      OutputTimeout = 250 // no more than once every 250 milliseconds
     };
      
   while(!FileIsEnding(handle))
     {
      // Read inputs
      CBufferType *in = new CBufferType();
      if(!in || !in.BufferInit(NeuronsToBar * BarsToLine, 1))
        {
         Print("Failed to initialize input buffer");
         delete in;
         FileClose(handle);
         return false;
        }
        
      for(int i = 0; i < NeuronsToBar * BarsToLine; i++)
         in.Update(i, (TYPE)FileReadNumber(handle));
         
      if(!data.Add(in))
        {
         Print("Failed to add input buffer to array");
         delete in;
         FileClose(handle);
         return false;
        }
        
      // Read targets
      CBufferType *out = new CBufferType();
      if(!out || !out.BufferInit(2, 1)) // 2 outputs: Buy, Sell
        {
         Print("Failed to initialize target buffer");
         delete out;
         FileClose(handle);
         return false;
        }
        
      // Read 2 targets from file (Direction, Distance)
      double t1 = FileReadNumber(handle);
      double t2 = FileReadNumber(handle);
      
      // Map to 2 outputs: Buy (0), Sell (1)
      // t1 is 1.0 for Buy, -1.0 for Sell
      
      out.Update(0, t1 > 0 ? 1.0 : 0.0); // Buy
      out.Update(1, t1 < 0 ? 1.0 : 0.0); // Sell
         
      if(!targets.Add(out))
        {
         Print("Failed to add target buffer to array");
         delete out;
         FileClose(handle);
         return false;
        }
        
      //--- output download progress in chart comment
      if(GetTickCount() > next_comment_time)
        {
         Comment(StringFormat("Loading data: %d samples", data.Total()));
         next_comment_time = GetTickCount() + OutputTimeout;
        }
     }
     
   Print("Loaded ", data.Total(), " samples.");
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
//| Train the model                                                  |
//+------------------------------------------------------------------+
bool NetworkFit(CNet &net, CArrayObj &data, CArrayObj &targets, VECTOR &loss_history, VECTOR &accuracy_history)
  {
   int total_data = data.Total();
   if(total_data == 0) return false;
   
   Print("Starting training loop...");
   
   uint next_comment_time = 0;
   enum { OutputTimeout = 250 };
   
   CEarlyStopping early_stopping(20, 0.001); // Patience 20 epochs, min_delta 0.1%

   for(int epoch = 0; epoch < Epochs; epoch++)
     {
      double epoch_loss = 0;
      int correct = 0;
      
      net.TrainMode(true);
      int batch_count = 0;
      
      for(int i = 0; i < total_data; i++)
        {
         CBufferType *in = data.At(i);
         CBufferType *target = targets.At(i);
         
         if(!net.FeedForward(in)) {
            Print("FeedForward failed at epoch ", epoch, " sample ", i);
            return false;
         }
         if(!net.Backpropagation(target)) {
            Print("Backpropagation failed at epoch ", epoch, " sample ", i);
            return false;
         }
         
         batch_count++;
         
         // Update weights every BatchSize samples (batch accumulation)
         if(batch_count >= BatchSize || i == total_data - 1)
           {
            if(!net.UpdateWeights(batch_count)) {
               Print("UpdateWeights failed at epoch ", epoch, " sample ", i);
               return false;
            }
            batch_count = 0;
           }
         
         epoch_loss += net.GetRecentAverageLoss();
         
         // Calculate accuracy (simple max index check)
         CBufferType *result = NULL;
         net.GetResults(result);
         if(result)
           {
            int pred_idx = 0;
            double max_val = result.At(0);
            for(int k = 1; k < (int)result.Total(); k++)
               if(result.At(k) > max_val) { max_val = result.At(k); pred_idx = k; }
               
            int target_idx = 0;
            max_val = target.At(0);
            for(int k = 1; k < (int)target.Total(); k++)
               if(target.At(k) > max_val) { max_val = target.At(k); target_idx = k; }
               
            if(pred_idx == target_idx) correct++;
           }
           
         // Update chart comment
         if(GetTickCount() > next_comment_time) {
            double curr_acc = (i > 0) ? (double)correct / (i + 1) * 100.0 : 0.0;
            double curr_loss = (i > 0) ? epoch_loss / (i + 1) : 0.0;
            Comment(StringFormat("Epoch %d/%d: Sample %d/%d (%.1f%%)\nLoss: %.5f\nAccuracy: %.2f%%", 
                                 epoch+1, Epochs, i+1, total_data, (double)(i+1)/total_data*100.0, curr_loss, curr_acc));
            next_comment_time = GetTickCount() + OutputTimeout;
         }
        }
        
      loss_history[epoch] = epoch_loss / total_data;
      accuracy_history[epoch] = (double)correct / total_data;
      
      PrintFormat("Epoch %d: Loss = %.5f, Accuracy = %.2f%%", epoch, loss_history[epoch], accuracy_history[epoch] * 100);
      Comment(StringFormat("Epoch %d/%d Completed.\nLoss: %.5f\nAccuracy: %.2f%%", 
                           epoch+1, Epochs, loss_history[epoch], accuracy_history[epoch] * 100));

      // Check for early stopping
      if(early_stopping.Check(loss_history[epoch]))
        {
         PrintFormat("Early stopping triggered at epoch %d", epoch);
         break;
        }
     }
     
   return true;
  }

//+------------------------------------------------------------------+
//| Save loss history                                                |
//+------------------------------------------------------------------+
void SaveLossHistory(string file_name, VECTOR &history)
  {
   int handle = FileOpen(file_name, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle != INVALID_HANDLE)
     {
      for(int i = 0; i < (int)history.Size(); i++)
         FileWrite(handle, i, history[i]);
      FileClose(handle);
     }
  }
