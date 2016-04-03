//-----------------------------------------------------------------------------
// Copyright 2016 Chuck Seberino
//
// This file is part of CCT.
//
// CCT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CCT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CCT.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------
#include "streamsyncTest.hpp"

using namespace CCT;

int StreamsyncTest::numStreams = 4;
int StreamsyncTest::numJobs = 20;
bool StreamsyncTest::syncStreams = true;

//-----------------------------------------------------------------------------
StreamsyncTest::StreamsyncTest(int index)
    : TestClass(index)
{
}

//-----------------------------------------------------------------------------
void StreamsyncTest::run()
{
    // Perform copy and compute test
    gpuPtr->timerStart(EventStream);
    int numIterations = IDIVUP(numJobs, numStreams);
    for (int n = 0; n < numIterations; ++n)
    {
        // Perform initial work as separate streams
        for (int ii = 0; ii < numStreams; ++ii)
        {
            // Wait for previous loop main stream
            gpuPtr->streamWait(ii, EventStream);
            // "Compute"
            gpuPtr->sleep(10+ii*50, ii);
            // Create event record for stream ii
            gpuPtr->timerStop(ii);

            // Break synchronization on last stream 
            if (syncStreams || ii != 3) 
            {
                // Tell main stream to wait for stream ii stop record.
                gpuPtr->streamWait(EventStream, ii);
            }
        }

        // Main stream "Compute"
        gpuPtr->sleep(100, EventStream);
        // Synchronization point for other streams
        gpuPtr->timerStop(EventStream);

        // Perform additional work as individual streams
        for (int ii = 0; ii < numStreams; ++ii)
        {
            // Wait for main stream to be complete.
            gpuPtr->streamWait(ii, EventStream);
            // "Compute"
            gpuPtr->sleep(30+10*ii, ii);
            // Create event record for stream ii
            gpuPtr->timerStop(ii);
            // Tell main stream to wait for stream ii stop record.
            gpuPtr->streamWait(EventStream, ii);
        }

        // Again, consolidate and run on a single stream
        gpuPtr->sleep(100, EventStream);
        // Synchronization point for other streams
        gpuPtr->timerStop(EventStream);
    }

    // Flush all outstanding work to ensure valid timing value.
    gpuPtr->timerStop(EventStream);
    gpuPtr->deviceSynchronize();
    CCT_INFO("Total time = " << gpuPtr->timerElapsed(EventStream) << "ms");
}
