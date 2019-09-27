// Andrew Lee
// Doing Marvin's Bullshit
// because this beats writing
// God, I hate writing...

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <future>
#include <limits>

constexpr int NUM_COLS = 10;
constexpr int protocols[] = { 1, 6, 17 };
std::string fakedirname, realdirname, realfile; // these shouldn't really be global but screw it

class sample
{
public:
   std::string toString()
   {
      std::string str = "";
      str += std::to_string(duration) + " ";
      str += std::to_string(src_dev) + " ";
      str += std::to_string(dst_dev) + " ";
      str += std::to_string(protocol) + " ";
      str += std::to_string(src_port) + " ";
      str += std::to_string(dst_port) + " ";
      str += std::to_string(src_packets) + " ";
      str += std::to_string(dst_packets) + " ";
      str += std::to_string(src_bytes) + " ";
      str += std::to_string(dst_bytes) + " ";
      return str;
   }
   int duration = 0;
   int src_dev = 0;
   int dst_dev = 0;
   int protocol = 0;
   int src_port = 0;
   int dst_port = 0;
   int src_packets = 0;
   int dst_packets = 0;
   int src_bytes = 0;
   int dst_bytes = 0;
};

std::pair<sample, sample> load_real_sample(int sample_length = 100, int random_state = 69);
void generate_random_samples(int random_state = 69, int sample_length = 100, int num_samples = 1000);

int main(int argc, char* argv[])
{
   std::cout << "Input fake directory name: ";
   std::cin >> fakedirname;
   std::cout << "Input real directory name: ";
   std::cin >> realdirname;
   std::cout << "Input real file name: ";
   std::cin >> realfile;
   auto f1 = std::async(&generate_random_samples, 69, 100, 10'000);
   auto f2 = std::async(&generate_random_samples, 69, 1'000, 10'000);
   auto f3 = std::async(&generate_random_samples, 69, 10'000, 2'000);
   auto f4 = std::async(&generate_random_samples, 69, 100'000, 1'160);
   f1.get(); f2.get(); f3.get(); f4.get(); // wait for all threads to finish
   std::cout << "All done!\n";
   return 0;
}

// you only use the min and max values, so we can store them in a "min sample" and "max sample" to mega decrease memory usage
std::pair<sample, sample> load_real_sample(int sample_length, int random_state)
{
   srand(random_state);
   std::string data_dir = "samples_" + std::to_string(sample_length);
   int sample_range = 10'000;
   if (sample_length == 100'000)
      sample_range = 1'160;
   else if (sample_length > 10'000)
      sample_range = 2000;
   int random_sample = rand() % sample_range;
   sample min, max;
   min.duration = min.src_dev = min.dst_dev = std::numeric_limits<int>::max();
   std::ifstream infile;
   std::string filename = realdirname + data_dir + realfile + "_sample_" + std::to_string(random_sample) + ".txt";
   infile.open(filename, std::ios::in);
   while (!infile.eof())
   {
      sample temp;
      infile >> temp.duration >> temp.src_dev >> temp.dst_dev >> temp.protocol >> temp.src_port >> temp.dst_port
         >> temp.src_packets >> temp.dst_packets >> temp.src_bytes >> temp.dst_bytes;
      if (temp.duration > max.duration)
         max.duration = temp.duration;
      if (temp.src_dev > max.src_dev)
         max.src_dev = temp.src_dev;
      if (temp.dst_dev > max.dst_dev)
         max.dst_dev = temp.dst_dev;
      if (temp.src_packets > max.src_packets)
         max.src_packets = temp.src_packets;
      if (temp.dst_packets > max.dst_packets)
         max.dst_packets = temp.dst_packets;
      if (temp.src_bytes > max.src_bytes)
         max.src_bytes = temp.src_bytes;
      if (temp.dst_bytes > max.dst_bytes)
         max.dst_bytes = temp.dst_bytes;
      if (temp.duration < min.duration)
         min.duration = temp.duration;
      if (temp.src_dev < min.src_dev)
         min.src_dev = temp.src_dev;
      if (temp.dst_dev < min.dst_dev)
         min.dst_dev = temp.dst_dev;
   }
   return std::make_pair(min, max);
}

void generate_random_samples(int random_state, int sample_length, int num_samples)
{
   srand(random_state);
   std::vector<sample> generated_samples;
   generated_samples.reserve(num_samples); // should speed up execution by minimizing alloc time
   for (int i = 0; i < num_samples; i++)
   {
      auto minmax = load_real_sample(sample_length, rand());
      sample min = std::get<0>(minmax);
      sample max = std::get<0>(minmax);
      for (int j = 0; j < sample_length; j++)
      {
         sample s;
         s.duration = rand() % (max.duration - min.duration) + min.duration;
         s.src_dev = rand() % (max.src_dev - min.src_dev) + min.src_dev;
         s.dst_dev = rand() % (max.dst_dev - min.dst_dev) + min.dst_dev;
         s.protocol = protocols[rand() % 3];
         s.src_port = rand() % 65535 + 1;
         s.dst_port = rand() % 65535 + 1;
         s.src_packets = rand() % max.src_packets;
         s.dst_packets = rand() % max.dst_packets;
         s.src_bytes = rand() % max.src_bytes;
         s.dst_bytes = rand() % max.dst_bytes;
         generated_samples.push_back(s);
      }
      std::ofstream outfile;
      std::string fakefilename = fakedirname + "samples_" + std::to_string(sample_length) + realfile + "_" + "random_sample_" + std::to_string(i) + ".txt";
      outfile.open(fakefilename, std::ofstream::out);
      for (auto& s:generated_samples)
      {
         outfile << s.toString() << "\n";
      }
      outfile.close();
      generated_samples.clear();
   }
   std::cout << "Finished generating " << num_samples << " samples of length " << sample_length << "\n";
}
